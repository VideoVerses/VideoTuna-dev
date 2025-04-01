import torch
import logging
import os
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys

from videotuna.flow.generation_base import GenerationFlow
from videotuna.utils.common_utils import instantiate_from_config
import videotuna.wan.wan as wan
from videotuna.wan.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from videotuna.wan.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from videotuna.wan.wan.utils.utils import cache_video, cache_image, str2bool

mainlogger = logging.getLogger("mainlogger")
EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

class WanVideoModelFlow(GenerationFlow):
    """
    Training and inference flow for YourModel.
    
    This model inherits from GenerationFlow, which is a base class for all generative models.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        task: str = "t2v-14B",             # choices: list(WAN_CONFIGS.keys())
        ckpt_dir: Optional[str] = None,    # Path to the checkpoint directory
        offload_model: Optional[bool] = None,  # Offload model to CPU after each forward pass
        ulysses_size: int = 1,             # Size of the ulysses parallelism in DiT
        ring_size: int = 1,                # Size of the ring attention parallelism in DiT
        t5_fsdp: bool = False,             # Whether to use FSDP for T5
        t5_cpu: bool = False,              # Whether to place T5 model on CPU
        dit_fsdp: bool = False,            # Whether to use FSDP for DiT
        save_file: Optional[str] = None,   # File path to save the generated media
        use_prompt_extend: bool = False,   # Whether to use prompt extend
        prompt_extend_method: str = "local_qwen",  # choices: ["dashscope", "local_qwen"]
        prompt_extend_model: Optional[str] = None,  # The prompt extend model to use
        prompt_extend_target_lang: str = "zh",      # choices: ["zh", "en"]
        base_seed: int = -1,               # Seed for generation
        *args, **kwargs
    ):
        super().__init__(
            first_stage_config,
            cond_stage_config,
            denoiser_config,
            scheduler_config,
            lr_scheduler_config,
        )

        self.task = task
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.base_seed = base_seed
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        if offload_model is None:
            offload_model = False if world_size > 1 else True
            logging.info(
                f"offload_model is not specified, set to {offload_model}.")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
        
        if ulysses_size > 1 or ring_size > 1:
            assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )

        if use_prompt_extend:
            if prompt_extend_method == "dashscope":
                self.prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model, is_vl="i2v" in task)
            elif prompt_extend_method == "local_qwen":
                self.prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    is_vl="i2v" in task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {prompt_extend_method}")

        cfg = WAN_CONFIGS[task]
        if ulysses_size > 1:
            assert cfg.num_heads % ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{ulysses_size=}`."

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            base_seed = base_seed[0]
        

        if "t2v" in task or "t2i" in task:
            logging.info("Creating WanT2V pipeline.")
            self.wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
                first_stage_model=self.first_stage_model,
                cond_stage_model=self.cond_stage_model,
                denoiser=self.denoiser
            )
        else:
            logging.info("Creating WanI2V pipeline.")
            self.wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
            )
        
    
    def forward(self, x, c, **kwargs):
        pass
    
    def training_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def inference(self, args):
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        
        prompt = None
        if "t2v" in self.task or "t2i" in self.task:
            if prompt is None:
                prompt = EXAMPLE_PROMPT[self.task]["prompt"]
            logging.info(f"Input prompt: {prompt}")
            if self.use_prompt_extend:
                logging.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt,
                        tar_lang=self.prompt_extend_target_lang,
                        seed=self.base_seed)
                    if prompt_output.status == False:
                        logging.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logging.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logging.info(f"Extended prompt: {prompt}")

            logging.info(
                f"Generating {'image' if 't2i' in self.task else 'video'} ...")
            video = self.wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args['size']],
                frame_num=args['frame_num'],
                shift=args['sample_shift'],
                sample_solver=args['sample_solver'],
                sampling_steps=args['sample_steps'],
                guide_scale=args['sample_guide_scale'],
                seed=self.base_seed,
                offload_model=self.offload_model)

        else:
            if prompt is None:
                prompt = EXAMPLE_PROMPT[self.task]["prompt"]
            if image is None:
                image = EXAMPLE_PROMPT[self.task]["image"]
            logging.info(f"Input prompt: {prompt}")
            logging.info(f"Input image: {image}")

            img = Image.open(image).convert("RGB")
            if self.use_prompt_extend:
                logging.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt,
                        tar_lang=self.prompt_extend_target_lang,
                        image=img,
                        seed=self.base_seed)
                    if prompt_output.status == False:
                        logging.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logging.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logging.info(f"Extended prompt: {prompt}")


            logging.info("Generating video ...")
            video = self.wan_i2v.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=self.base_seed,
                offload_model=self.offload_model)

        save_file = None
        if rank == 0:
            if save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in self.task else '.mp4'
                save_file = f"{self.task}_{size.replace('*','x') if sys.platform=='win32' else args['size']}_{self.ulysses_size}_{self.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

            if "t2i" in self.task:
                logging.info(f"Saving generated image to {save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=save_file,
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
        logging.info("Finished.")
    

    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None):
        pass