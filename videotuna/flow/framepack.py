import torch
from loguru import logger
import random
import os
import math
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys
from omegaconf import OmegaConf, DictConfig
import numpy as np
from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.args_utils import VideoMode

# framepack specific
from videotuna.models.framepack.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from videotuna.models.framepack.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop
from videotuna.models.framepack.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from videotuna.models.framepack.pipelines.k_diffusion_hunyuan import sample_hunyuan
from videotuna.models.framepack.memory import (
    cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation, fake_diffusers_current_device,
    DynamicSwapInstaller, unload_complete_models, load_model_as_complete
)
from videotuna.models.framepack.thread_utils import AsyncStream, async_run
from transformers import SiglipImageProcessor, SiglipVisionModel
from videotuna.models.framepack.clip_vision import hf_clip_vision_encode
from videotuna.models.framepack.bucket_tools import find_nearest_bucket
import traceback
from videotuna.schedulers.flow_matching import FlowMatchScheduler

# wan specific 
import videotuna.models.wan.wan as wan
from videotuna.models.wan.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from videotuna.models.wan.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from videotuna.models.wan.wan.utils.utils import cache_video, cache_image, str2bool

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
            "inputs/i2v/576x1024/i2v_input.JPG",
    },
}

class HunyuanVideoPackedFlow(GenerationBase):
    """
    Training and inference flow for YourModel.
    
    This model inherits from GenerationFlow, which is a base class for all generative models.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        tokenizer_config_2: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        feature_extractor_config: Optional[Dict[str, Any]] = None,
        image_encoder_config: Optional[Dict[str, Any]] = None,
        high_vram: bool = False,           
        latent_window_size: int = 9,
        # below are probably not needed
        # task: str = "t2v-14B",            
        ckpt_path: Optional[str] = None,    
        offload_model: Optional[bool] = None, 
        ulysses_size: int = 1,             
        ring_size: int = 1,               
        t5_fsdp: bool = False,           
        t5_cpu: bool = False,             
        dit_fsdp: bool = False,            
        use_prompt_extend: bool = False,   
        prompt_extend_method: str = "local_qwen",  
        prompt_extend_model: Optional[str] = None,  
        prompt_extend_target_lang: str = "zh",    
        seed: int = -1,
        *args, **kwargs
    ):
        logger.info("HunyuanVideoPackedFlow flow: starting init")
        assert ckpt_path is not None, "Please specify the checkpoint directory."
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            lora_config=lora_config,
            trainable_components=[]
        )
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device_id = local_rank
        self.device_id = device_id
        device = torch.device(f"cuda:{self.device_id}")

        self.feature_extractor = instantiate_from_config(feature_extractor_config)
        self.image_encoder = instantiate_from_config(image_encoder_config)
        self.image_encoder.eval()
        
        self.image_encoder.to(device) # TODO: check if this is correct
        self.image_encoder.requires_grad_(False)
        self.tokenizer = instantiate_from_config(tokenizer_config)
        self.tokenizer_2 = instantiate_from_config(tokenizer_config_2)
        logger.info("HunyuanVideoPackedFlow flow: class init finished")
        self.ckpt_path = ckpt_path
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.seed = seed
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.high_vram = high_vram
        self.latent_window_size = latent_window_size

        # exit()

    def _validate_args(self, args):        
        # Size reassign and check
        args.size = f"{args.width}*{args.height}"
        logger.info(f"setting size = width*height == {args.size}")
        assert args.size in SUPPORTED_SIZES[
            self.task], f"Unsupport size {args.size} for task {self.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[self.task])}"

    def generate_timestamp(self):
        now = datetime.now()
        timestamp = now.strftime('%y%m%d_%H%M%S')
        milliseconds = f"{int(now.microsecond / 1000):03d}"
        random_number = random.randint(0, 9999)
        return f"{timestamp}_{milliseconds}_{random_number}"
    
    @torch.no_grad()
    def inference(self, args: DictConfig): 
        seed = args.seed
        video_length_in_second = args.video_length_in_second
        latent_window_size = args.latent_window_size
        steps = args.steps
        cfg = args.cfg
        gs = args.gs
        rs = args.rs
        gpu_memory_preservation = args.gpu_memory_preservation
        use_teacache = args.use_teacache
        mp4_crf = args.mp4_crf
        prompt = args.prompt
        n_prompt = args.n_prompt
        input_image_path = args.input_image_path
        height = args.height
        width = args.width
        outputs_folder = './outputs/'
        os.makedirs(outputs_folder, exist_ok=True)
        
        self.denoiser.to(dtype=torch.bfloat16)
        self.first_stage_model.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.cond_stage_model.to(dtype=torch.float16)
        self.cond_stage_2_model.to(dtype=torch.float16)

        print(f'self.denoiser.dtype = {self.denoiser.dtype}')
        # exit()

        self.denoiser.cuda()
        self.cond_stage_model.cuda()
        self.cond_stage_2_model.cuda()
        self.image_encoder.cuda()
        self.first_stage_model.cuda()


        input_image = np.array(Image.open(input_image_path))

        num_latent_sections = (video_length_in_second * 30) / (latent_window_size * 4)
        num_latent_sections = int(max(round(num_latent_sections), 1))

        job_id = self.generate_timestamp()
        
        print("Starting video generation...")

        try:
            # Clean GPU
            if not self.high_vram:
                unload_complete_models(
                    self.cond_stage_model, self.cond_stage_2_model, self.image_encoder, self.first_stage_model, self.denoiser
                )

            # Text encoding
            print("Text encoding...")

            if not self.high_vram:
                fake_diffusers_current_device(self.cond_stage_model, gpu)
                load_model_as_complete(self.cond_stage_2_model, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.cond_stage_model, self.cond_stage_2_model, self.tokenizer, self.tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.cond_stage_model, self.cond_stage_2_model, self.tokenizer, self.tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Load input image
            print("Image processing...")

            H, W, C = input_image.shape
            # height, width = find_nearest_bucket(H, W, resolution=640) # NOTE load from args
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            # Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # encode the input image
            print("VAE encoding...")

            if not self.high_vram:
                load_model_as_complete(self.first_stage_model, target_device=gpu)

            start_latent = vae_encode(input_image_pt, self.first_stage_model)
            print(f'start_latent.shape = {start_latent.shape}') # [1, 16, 1, 80, 76]

            # extract the CLIP input image features
            print("CLIP Vision encoding...")

            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype
            llama_vec = llama_vec.to(self.denoiser.dtype)
            llama_vec_n = llama_vec_n.to(self.denoiser.dtype)
            clip_l_pooler = clip_l_pooler.to(self.denoiser.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.denoiser.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.denoiser.dtype)

            # Sampling
            print("Start sampling...")

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3
            
            # 19 -> 28 -> 37 -> 46 -> 55
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            print(f'history_latents.shape = {history_latents.shape}') # [1, 16, 19, 80, 76], 19 = 18 + 1
            history_rgbs = None
            num_generated_latent_frames = 0

            latent_paddings = reversed(range(num_latent_sections))

            if num_latent_sections > 4:
                # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
                # items looks better than expanding it when num_latent_sections > 4
                print(f'num_latent_sections = {num_latent_sections}')
                latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]

            final_output_filename = None
            
            for latent_section_index in latent_paddings:
                # seems latent_padding should be named as latent_section_index
                print(f'current latent_section_index = {latent_section_index}')
                is_last_section = latent_section_index == 0
                # 27, 18, 9, 0
                latent_padding_size = latent_section_index * latent_window_size

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}') # 27

                # first frame, 27, 9, 1, 2, 16 ??? 
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                print(f'indices = {indices}') # [0,1,2,...,55]
                clean_latent_indices_pre, blank_indices, cur_latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                print(f'clean_latent_indices_pre = {clean_latent_indices_pre}') # [0]
                print(f'blank_indices = {blank_indices}') # [1,2,...,27]
                print(f'cur_latent_indices = {cur_latent_indices}') # [28,29,...,36]
                print(f'clean_latent_indices_post = {clean_latent_indices_post}') # [37]
                print(f'clean_latent_2x_indices = {clean_latent_2x_indices}') # [38,39]
                print(f'clean_latent_4x_indices = {clean_latent_4x_indices}') # [40,41,...,55]
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                print(f'clean_latent_indices = {clean_latent_indices}') # [0,37]
                # exit()

                # start_latent: the latent of the input image
                clean_latents_pre = start_latent.to(history_latents)
                # post: 0th frame, 2x: 1-2 frame, 4x: 3-18 frame
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                print(f'clean_latents = {clean_latents.max()}, {clean_latents.min()}')

                # exit()
                
                
                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.denoiser, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    self.denoiser.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.denoiser.initialize_teacache(enable_teacache=False)

                def callback(d):
                    current_step = d['i'] + 1
                    print(f"Sampling step {current_step}/{steps}")
                    return

                generated_latents = sample_hunyuan(
                    transformer=self.denoiser,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    # dtype=torch.float32,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=cur_latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
                print(f'generated_latents.shape = {generated_latents.shape}') # [1, 16, 9, 80, 76]
                print(f'generated_latents = {generated_latents.max()}, {generated_latents.min()}')

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                num_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                print(f'history_latents.shape = {history_latents.shape}') # [1, 16, 19, 80, 76]

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.denoiser, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.first_stage_model, target_device=gpu)

                valid_history_latents = history_latents[:, :, :num_generated_latent_frames, :, :]
                print(f'valid_history_latents.shape = {valid_history_latents.shape}') # [1, 16, 18, 80, 76][1, 16, 27, 80, 76] [1, 16, 37, 80, 76]

                # VAE decoding
                if history_rgbs is None:
                    history_rgbs = vae_decode(valid_history_latents, self.first_stage_model).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_rgbs = vae_decode(valid_history_latents[:, :, :section_latent_frames], self.first_stage_model).cpu()
                    history_rgbs = soft_append_bcthw(current_rgbs, history_rgbs, overlapped_frames)

                if not self.high_vram:
                    unload_complete_models()

                # save the generated video
                output_filename = os.path.join(outputs_folder, f'{job_id}_{num_generated_latent_frames}.mp4')
                final_output_filename = output_filename

                print(f'history_rgbs.shape = {history_rgbs.shape}') # [1, 3, 145, 640, 608]
                save_bcthw_as_mp4(history_rgbs, output_filename, fps=30, crf=mp4_crf)

                print(f'Decoded. Current latent shape {valid_history_latents.shape}; pixel shape {history_rgbs.shape}')

                if is_last_section:
                    break
                    
            
            return final_output_filename
        
        except Exception as e:
            traceback.print_exc()

            if not self.high_vram:
                unload_complete_models(
                    self.cond_stage_model, self.cond_stage_2_model, self.image_encoder, self.first_stage_model, self.denoiser
                )
            
            return None

    # def load_weight(self):
        # self.cond_stage_model.load_weight()
        # self.cond_stage_2_model.load_weight()
        # self.first_stage_model.load_weight()
        # self.feature_extractor.load_weight()
        # self.image_encoder.load_weight()
        #denoiser use from_pretrained, no need load again

        # self.cond_stage_model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        # self.cond_stage_2_model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        # self.first_stage_model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        # self.feature_extractor.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        # self.image_encoder.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        # self.load_first_stage(self.ckpt_path, ignore_missing_ckpts=True)
        # self.load_cond_stage(self.ckpt_path, ignore_missing_ckpts=True)
        # self.load_cond_stage_2(self.ckpt_path, ignore_missing_ckpts=True)

        # if dist.is_initialized():
        #     dist.barrier()
        # if self.dit_fsdp:
        #     self.model = self.shard_fn(self.model)
        # else:
        #     if not self.init_on_cpu:
        #         self.model = self.model.to(self.device)
        # pass
    
    def load_denoiser(self, ckpt_path: str = None, denoiser_ckpt_path: str = None, ignore_missing_ckpts: bool = False):
        # return super().load_denoiser(ckpt_path, denoiser_ckpt_path, ignore_missing_ckpts)
        ckpt_path = os.path.join(ckpt_path, denoiser_ckpt_path)
        # self.denoiser.load_state_dict(torch.load(ckpt_path, map_location='cpu'))     
        # self.denoiser = self.load_model(self.denoiser, ckpt_path, strict=False)
        self.denoiser = self.load_model(self.denoiser, ckpt_path)


    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False):
        # if "t2v" in self.task or "t2i" in self.task:
        #     self.wan_t2v.load_weight()
        #     #this is only used to load trained denoiser_ckpt_path, 
        #     #so we set ignore missing ckpts avoid duplicate loading
        #     self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
        # else:
        #     self.wan_i2v.load_weight()
        #     self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
        # pass
        # self.load_weight()
        # self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
        pass

    
    def enable_vram_management(self):
        pass
    
    def vae_encode_video(self, videos):
        latents = []
        for idx in range(videos.shape[2]):
            latents.append(vae_encode(videos[:, :, idx:idx+1, :, :], self.first_stage_model))
        return torch.cat(latents, dim=2)
    
    def training_step(self, batch, batch_idx):
        # 1. get first frame
        videos = batch[self.first_stage_key]
        total_frames = videos.shape[2]
        first_frame = videos[:, :, 0:1, :, :]

        # TODO 
        model_offload = False
        device = "cuda"
        dtype = torch.bfloat16
        cfg = 1
        n_prompt = ""
        device = torch.device(f"cuda:{self.device_id}")

        with torch.no_grad():
            if model_offload:
                self.vae.model.to(device)
                latents = torch.stack(self.vae.encode(videos)).to(dtype=dtype, device=device).detach()
                videos[:, :, 1:, :, :] = 0
                y = torch.stack(self.vae.encode(videos)).to(dtype=dtype, device=device).detach()
                self.vae.model.to('cpu')
                self.text_encoder.model.to(device)
                text_cond_embed = self.text_encoder(batch[self.cond_stage_key], device)
                self.text_encoder.model.to('cpu')
                self.clip.model.to(device)
                clip_context = self.clip.visual(first_frame)
                self.clip.model.to('cpu')
            else:
                # TODO check 
                latents = self.vae_encode_video(videos).to(dtype=dtype, device=device).detach()

                videos[:, :, 1:, :, :] = 0
                y = self.vae_encode_video(videos).to(dtype=dtype, device=device).detach()

                start_latent = vae_encode(first_frame, self.first_stage_model)
                # TODO not elegant
                first_frame_np = first_frame[0, :, 0, :, :].to(torch.float32).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                image_encoder_output = hf_clip_vision_encode(first_frame_np, self.feature_extractor, self.image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

                # text encoding
                llama_vec, clip_l_pooler = encode_prompt_conds(batch[self.cond_stage_key][0], 
                                                               self.cond_stage_model, self.cond_stage_2_model, 
                                                               self.tokenizer, self.tokenizer_2)

                if cfg == 1:
                    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                else:
                    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, 
                                                                      self.cond_stage_model, self.cond_stage_2_model, 
                                                                      self.tokenizer, self.tokenizer_2)

                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # exit()
        ## scheduler TODO this should be moved to init 
        self.scheduler : FlowMatchScheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        ## noise
        # 29 frames in total; 0th: input frame; 1-9: noisy frames; 10: clean frame; 11-12: 2x frames; 13-28: 4x frames
        num_frames = self.latent_window_size * 4 - 3
        num_noisy_frames = (num_frames + 3) // 4
        noisy_latent_indices = list(range(total_frames))[1:num_noisy_frames+1]
        b, c, f, h, w = latents.shape
        noise = torch.randn_like(latents[:, :, :num_noisy_frames, :, :])
        timestep_ids = torch.randint(0, self.scheduler.num_train_timesteps, (b,))
        timesteps = self.scheduler.timesteps[timestep_ids].to(dtype=dtype, device=device)
        # timesteps = self.scheduler.timesteps[timestep_ids].to(dtype=dtype)
        noisy_latents = self.scheduler.add_noise(latents[:, :, 1:num_noisy_frames+1, :, :], noise, timesteps).to(dtype=dtype, device=device)
        # noisy_latents = self.scheduler.add_noise(latents[:, :, 1:num_noisy_frames+1, :, :], noise, timesteps).to(dtype=dtype)
        training_target = noise.to(device) - latents[:, :, 1:num_noisy_frames+1, :, :]
        # training_target = noise - latents[:, :, 1:num_noisy_frames+1, :, :]


        # split indices for different latent sections
        indices = torch.arange(0, sum([1, 0, self.latent_window_size, 1, 2, 16])).unsqueeze(0)
        splits = [1, 0, self.latent_window_size, 1, 2, 16]
        clean_latent_indices_pre, blank_indices, cur_latent_indices, \
            clean_latent_indices_post, clean_latent_2x_indices, \
            clean_latent_4x_indices = indices.split(splits, dim=1)

        # get condition latents: clean_latents, clean_latent_2x, clean_latent_4x
        clean_latents_pre = start_latent.to(noisy_latents)
        # post: 0th frame, 2x: 1-2 frame, 4x: 3-18 frame
        clean_latents_post = latents[:, :, 10:11, :, :]
        clean_latents_2x = latents[:, :, 11:13, :, :]
        clean_latents_4x = latents[:, :, 13:29, :, :]
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        distilled_guidance_scale = 10
        distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * b).to(device=device, dtype=dtype)
        # distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * b).to(dtype=dtype)

        noise_pred = self.denoiser(noisy_latents, timesteps, 
                                encoder_hidden_states=llama_vec, 
                                encoder_attention_mask=llama_attention_mask,
                                pooled_projections=clip_l_pooler,
                                guidance=distilled_guidance,
                                latent_indices=cur_latent_indices,
                                clean_latents=clean_latents,
                                clean_latent_indices=clean_latent_indices,
                                clean_latents_2x=clean_latents_2x,
                                clean_latent_2x_indices=clean_latent_2x_indices,
                                clean_latents_4x=clean_latents_4x,
                                clean_latent_4x_indices=clean_latent_4x_indices,
                                image_embeddings=image_encoder_last_hidden_state,
                                return_dict=False, # TODO 
                                )
        # print(f'noise_pred.shape = {noise_pred.shape}') # [1, 16, 9, 80, 76]
        loss = torch.nn.functional.mse_loss(torch.stack(noise_pred).float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timesteps).to(device=device)
        # loss = loss * self.scheduler.training_weight(timesteps)
        return loss
        
        
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        pass