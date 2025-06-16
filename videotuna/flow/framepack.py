import torch
from loguru import logger
import random
import os
import math
from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
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
import tempfile

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

# wan specific 
from videotuna.utils.quantization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

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
        torch_compile: bool = False,
        fp8_scale : bool = False,  
        cfg_parallel : bool = False,    
        # below are probably not needed
        # task: str = "t2v-14B",            
        ckpt_path: Optional[str] = None,    
        offload_model: Optional[bool] = None,        
        seed: int = -1,
        *args, **kwargs
    ):
        logger.info("HunyuanVideoPackedFlow flow: starting init")
        if cfg_parallel:
            dist.init_process_group("nccl")
            init_distributed_environment(
                rank=dist.get_rank(), 
                world_size=dist.get_world_size()
            )
            
            initialize_model_parallel(
                classifier_free_guidance_degree=2
            )
            torch.cuda.set_device(dist.get_rank())
        assert ckpt_path is not None, "Please specify the checkpoint directory."
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            # lora_config=lora_config,
            trainable_components=[]
        )
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        self.feature_extractor = instantiate_from_config(feature_extractor_config)
        self.image_encoder = instantiate_from_config(image_encoder_config)
        self.image_encoder.eval()
        
        self.image_encoder.to("cuda") # TODO: check if this is correct
        self.image_encoder.requires_grad_(False)
        self.tokenizer = instantiate_from_config(tokenizer_config)
        self.tokenizer_2 = instantiate_from_config(tokenizer_config_2)
        if fp8_scale:
            state_dict = self.denoiser.state_dict()
            state_dict = self.fp8_optimization(self.denoiser, state_dict, device, True, use_scaled_mm=False)
            info = self.denoiser.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded FP8 optimized weights: {info}")
        logger.info("HunyuanVideoPackedFlow flow: class init finished")
        self.ckpt_path = ckpt_path
        self.seed = seed
        self.offload_model = offload_model
        self.high_vram = False
        self.torch_compile = torch_compile
        self.fp8_scale = fp8_scale
        self.cfg_parallel = cfg_parallel


    def generate_timestamp(self):
        now = datetime.now()
        timestamp = now.strftime('%y%m%d_%H%M%S')
        milliseconds = f"{int(now.microsecond / 1000):03d}"
        random_number = random.randint(0, 9999)
        return f"{timestamp}_{milliseconds}_{random_number}"
    
    @torch.no_grad()
    def warmup(self, args:DictConfig):
        if not self.torch_compile:
            logger.info("Skipping warmup since no compile")
            return
        steps = args.num_inference_steps
        args.num_inference_steps = 1
        args.warmup = True
        self.inference(args)
        args.num_inference_steps = steps
        args.warmup = False
        logger.info("warmup finished")

    @torch.no_grad()
    def inference(self, args: DictConfig): 
        # 如果是批量生成转场视频
        if hasattr(args, 'batch_transition') and args.batch_transition:
            return self._batch_transition_inference(args)
        
        mode = args.mode
        seed = args.seed
        video_length_in_second = args.video_length_in_second
        latent_window_size = args.latent_window_size
        steps = args.num_inference_steps
        cfg = args.unconditional_guidance_scale
        gs = args.gs
        rs = args.rs
        gpu_memory_preservation = args.gpu_memory_preservation
        use_teacache = args.use_teacache
        mp4_crf = args.mp4_crf
        prompt = args.prompt
        n_prompt = args.n_prompt
        input_image_path = args.input_image_path
        warmup = args.get('warmup', False)
        outputs_folder = args.savedir
        device = args.get("device", "cuda")
        os.makedirs(outputs_folder, exist_ok=True)
        
        denoiser_dtype = torch.bfloat16
        input_image = np.array(Image.open(input_image_path))    
        end_image = None
        if mode == 'transition':
            end_image_path = args.end_image_path
            end_image = np.array(Image.open(end_image_path))

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
                #fake_diffusers_current_device(self.cond_stage_model, device)
                load_model_as_complete(self.cond_stage_model, target_device=device, unload=True)
                load_model_as_complete(self.cond_stage_2_model, target_device=device, unload=False)

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
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            end_image_np = None
            if end_image is not None:
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            end_image_pt = None
            if end_image_np is not None:
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

            # encode the input image
            print("VAE encoding...")

            if not self.high_vram:
                load_model_as_complete(self.first_stage_model, target_device=device)

            start_latent = vae_encode(input_image_pt, self.first_stage_model)
            print(f'start_latent.shape = {start_latent.shape}')
            end_latent = None
            if end_image_pt is not None:
                end_latent = vae_encode(end_image_pt, self.first_stage_model)
                print(f'end_latent.shape = {end_latent.shape}')
            
            # extract the CLIP input image features
            print("CLIP Vision encoding...")

            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=device)

            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype
            llama_vec = llama_vec.to(denoiser_dtype)
            llama_vec_n = llama_vec_n.to(denoiser_dtype)
            clip_l_pooler = clip_l_pooler.to(denoiser_dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(denoiser_dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(denoiser_dtype)

            # Sampling
            print("Start sampling...")

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3
            
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            print(f'history_latents.shape = {history_latents.shape}')
            if end_latent is not None:
                history_latents[:, :, 0:1, :, :] = end_latent
            history_rgbs = None
            num_generated_latent_frames = 0

            latent_paddings = reversed(range(num_latent_sections))

            if num_latent_sections > 4:
                print(f'num_latent_sections = {num_latent_sections}')
                latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]

            final_output_filename = None
            
            for latent_section_index in latent_paddings:
                print(f'current latent_section_index = {latent_section_index}')
                is_last_section = latent_section_index == 0
                latent_padding_size = latent_section_index * latent_window_size

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                print(f'indices = {indices}')
                clean_latent_indices_pre, blank_indices, cur_latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                print(f'clean_latent_indices_pre = {clean_latent_indices_pre}')
                print(f'blank_indices = {blank_indices}')
                print(f'cur_latent_indices = {cur_latent_indices}')
                print(f'clean_latent_indices_post = {clean_latent_indices_post}')
                print(f'clean_latent_2x_indices = {clean_latent_2x_indices}')
                print(f'clean_latent_4x_indices = {clean_latent_4x_indices}')
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                print(f'clean_latent_indices = {clean_latent_indices}')

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.denoiser, target_device=device, preserved_memory_gb=gpu_memory_preservation)

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
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=denoiser_dtype,
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
                print(f'generated_latents.shape = {generated_latents.shape}')

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                num_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                print(f'history_latents.shape = {history_latents.shape}')

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.denoiser, target_device=device, preserved_memory_gb=49)
                    load_model_as_complete(self.first_stage_model, target_device=device)

                valid_history_latents = history_latents[:, :, :num_generated_latent_frames, :, :]
                print(f'valid_history_latents.shape = {valid_history_latents.shape}')

                if history_rgbs is None:
                    history_rgbs = vae_decode(valid_history_latents, self.first_stage_model).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_rgbs = vae_decode(valid_history_latents[:, :, :section_latent_frames], self.first_stage_model).cpu()
                    history_rgbs = soft_append_bcthw(current_rgbs, history_rgbs, overlapped_frames)

                if not self.high_vram:
                    unload_complete_models()

                if not warmup and (not dist.is_initialized() or dist.get_rank() == 0):
                    output_filename = os.path.join(outputs_folder, f'{job_id}_{num_generated_latent_frames}.mp4')
                    final_output_filename = output_filename

                    print(f'history_rgbs.shape = {history_rgbs.shape}')
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

    def _batch_transition_inference(self, args: DictConfig):
        """
        批量生成转场视频的推理方法
        
        Args:
            args: 配置参数，包含：
                - input_dir: 输入视频目录
                - output_dir: 输出目录
                - transition_duration: 每个过渡部分的持续时间（默认2秒）
                - mp4_crf: 视频质量参数
                - openai_api_key: OpenAI API密钥
                - 其他framepack参数
        """
        input_dir = "transition-input"
        output_dir = "res"
        transition_duration = 2
        mp4_crf = 16
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有MP4文件并按数字大小排序
        mp4_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        mp4_files.sort(key=lambda x: int(x.split('.')[0]))  # 按文件名中的数字排序
        
        if len(mp4_files) < 2:
            logger.warning("至少需要2个MP4文件才能生成转场视频")
            return []
        
        transition_videos = []
        
        # 处理每对连续的视频
        for i in range(len(mp4_files) - 1):
            first_video = os.path.join(input_dir, mp4_files[i])
            second_video = os.path.join(input_dir, mp4_files[i+1])
            
            # 创建转场目录
            transition_dir = os.path.join(output_dir, f"transition_{i+1}_to_{i+2}")
            os.makedirs(transition_dir, exist_ok=True)
            
            # 创建res目录用于保存帧
            res_dir = os.path.join(transition_dir, "res")
            os.makedirs(res_dir, exist_ok=True)
            
            # 提取帧
            last_frame = self.extract_frame(first_video, -1)
            first_frame = self.extract_frame(second_video, 0)
            
            # 保存帧
            Image.fromarray(last_frame).save(os.path.join(res_dir, "last_frame.png"))
            Image.fromarray(first_frame).save(os.path.join(res_dir, "first_frame.png"))
            
            # 检查是否已存在过渡帧
            transition_frame_path = os.path.join(transition_dir, "transition_frame.png")
            if os.path.exists(transition_frame_path):
                logger.info(f"使用已存在的过渡帧: {transition_frame_path}")
                transition_frame = np.array(Image.open(transition_frame_path))
            else:
                # 生成中间帧
                logger.info(f"生成 {mp4_files[i]} 和 {mp4_files[i+1]} 之间的过渡帧")
                transition_frame = self.generate_transition_frame(last_frame, first_frame, openai_api_key)
                
                # 保存中间帧
                Image.fromarray(transition_frame).save(transition_frame_path)
                Image.fromarray(transition_frame).save(os.path.join(res_dir, "transition_frame.png"))
            
            # 生成第一段过渡视频（尾帧到中间帧）
            first_transition_path = os.path.join(transition_dir, "first_transition.mp4")
            logger.info(f"生成第一段过渡视频: {first_transition_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                input_image_path = os.path.join(temp_dir, "input_image.png")
                end_image_path = os.path.join(temp_dir, "end_image.png")
                
                # 保存图像
                Image.fromarray(last_frame).save(input_image_path)
                Image.fromarray(transition_frame).save(end_image_path)
                
                # 创建framepack配置
                first_args = OmegaConf.create({
                    "seed": args.get("seed", random.randint(0, 2**32 - 1)),
                    "video_length_in_second": transition_duration,
                    "latent_window_size": 9,
                    "num_inference_steps": args.get("num_inference_steps", 20),
                    "unconditional_guidance_scale": args.get("unconditional_guidance_scale", 7.5),
                    "gs": args.get("gs", 2.0),
                    "rs": args.get("rs", 0.5),
                    "gpu_memory_preservation": args.get("gpu_memory_preservation", 12),
                    "use_teacache": args.get("use_teacache", True),
                    "mp4_crf": mp4_crf,
                    "prompt": args.get("prompt", "A seamless transition between two images, smooth and natural camera movement."),
                    "n_prompt": args.get("n_prompt", "low quality, distortion, blurry, ugly"),
                    "input_image_path": input_image_path,
                    "end_image_path": end_image_path,
                    "savedir": os.path.dirname(first_transition_path)
                })
                
                # 生成第一段视频
                first_result = self.inference(first_args)
                
                if first_result and os.path.exists(first_result):
                    if first_result != first_transition_path:
                        import shutil
                        shutil.move(first_result, first_transition_path)
                else:
                    logger.error(f"生成第一段过渡视频失败")
                    continue
            
            # 生成第二段过渡视频（中间帧到首帧）
            second_transition_path = os.path.join(transition_dir, "second_transition.mp4")
            logger.info(f"生成第二段过渡视频: {second_transition_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                input_image_path = os.path.join(temp_dir, "input_image.png")
                end_image_path = os.path.join(temp_dir, "end_image.png")
                
                # 保存图像
                Image.fromarray(transition_frame).save(input_image_path)
                Image.fromarray(first_frame).save(end_image_path)
                
                # 创建framepack配置
                second_args = OmegaConf.create({
                    "seed": args.get("seed", random.randint(0, 2**32 - 1)),
                    "video_length_in_second": transition_duration,
                    "latent_window_size": 9,
                    "num_inference_steps": args.get("num_inference_steps", 20),
                    "unconditional_guidance_scale": args.get("unconditional_guidance_scale", 7.5),
                    "gs": args.get("gs", 2.0),
                    "rs": args.get("rs", 0.5),
                    "gpu_memory_preservation": args.get("gpu_memory_preservation", 12),
                    "use_teacache": args.get("use_teacache", True),
                    "mp4_crf": mp4_crf,
                    "prompt": args.get("prompt", "A seamless transition between two images, smooth and natural camera movement."),
                    "n_prompt": args.get("n_prompt", "low quality, distortion, blurry, ugly"),
                    "input_image_path": input_image_path,
                    "end_image_path": end_image_path,
                    "savedir": os.path.dirname(second_transition_path)
                })
                
                # 生成第二段视频
                second_result = self.inference(second_args)
                
                if second_result and os.path.exists(second_result):
                    if second_result != second_transition_path:
                        import shutil
                        shutil.move(second_result, second_transition_path)
                else:
                    logger.error(f"生成第二段过渡视频失败")
                    continue
            
            # 合并两段视频
            merged_transition_path = os.path.join(transition_dir, "merged_transition.mp4")
            logger.info(f"合并过渡视频: {merged_transition_path}")
            merged_path = self.merge_videos(
                [first_transition_path, second_transition_path],
                merged_transition_path,
                mp4_crf
            )
            
            if merged_path:
                transition_videos.append(merged_path)
                logger.info(f"生成转场视频: {merged_path}")
            else:
                logger.error(f"合并过渡视频失败")
        
        return transition_videos

    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False):
        pass
    
    def enable_vram_management(self):
        self.first_stage_model.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.cond_stage_model.to(dtype=torch.float16)
        self.cond_stage_2_model.to(dtype=torch.float16)

        self.cond_stage_model.cuda()
        self.cond_stage_2_model.cuda()
        self.image_encoder.cuda()
        self.first_stage_model.cuda()
        self.denoiser.cuda()
        if self.torch_compile:
            torch._dynamo.config.cache_size_limit = 64
            self.denoiser = torch.compile(self.denoiser, mode="max-autotune-no-cudagraphs")
        
    
    def training_step(self, batch, batch_idx):
        print('batch keys:', batch.keys())
        exit()
        
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        pass

    def fp8_optimization(
        self, model, state_dict: dict[str, torch.Tensor], device: torch.device, move_to_device: bool, use_scaled_mm: bool = False
    ) -> dict[str, torch.Tensor]:  # Return type hint added
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
            use_scaled_mm (bool):
                Whether to use scaled matrix multiplication for FP8.
        """
        TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
        EXCLUDE_KEYS = ["norm"]  # Exclude norm layers (e.g., LayerNorm, RMSNorm) from FP8

        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device)

        # apply monkey patching
        apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=use_scaled_mm)

        return state_dict

    def extract_frame(self, video_path: str, frame_index: int = -1) -> np.ndarray:
        """
        Extract a specific frame from a video file.
        
        Args:
            video_path: Path to the video file.
            frame_index: Index of the frame to extract. If -1, extracts the last frame.
            
        Returns:
            The extracted frame as a numpy array.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_index == -1:
            frame_index = total_frames - 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to extract frame {frame_index} from {video_path}")
        
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
        
    def generate_transition_frame(self, start_frame: np.ndarray, end_frame: np.ndarray, openai_api_key: Optional[str] = None) -> np.ndarray:
        """
        Generate a transition frame between two frames using OpenAI API (GPT-Image-1).
        
        Args:
            start_frame: The starting frame.
            end_frame: The ending frame.
            openai_api_key: OpenAI API key. If None, will try to get it from environment.
            
        Returns:
            The generated transition frame as a numpy array.
        """
        try:
            from openai import OpenAI
            import tempfile
            from io import BytesIO
            import requests
            
            # Get API key from environment if not provided
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not provided. Cannot generate transition frame.")
                # Return a simple blend of the two frames as a fallback
                return (start_frame.astype(np.float32) * 0.5 + end_frame.astype(np.float32) * 0.5).astype(np.uint8)
            
            # Initialize client
            client = OpenAI(api_key=api_key)
            
            # Convert frames to PIL Images
            start_img = Image.fromarray(start_frame)
            end_img = Image.fromarray(end_frame)
            
            # Save images to temporary files
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as start_file:
                start_img.save(start_file.name)
                start_file_path = start_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as end_file:
                end_img.save(end_file.name)
                end_file_path = end_file.name
            
            try:
                # Open the image files for GPT-Image-1
                image_files = []
                for image_path in [start_file_path, end_file_path]:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                        image_file = BytesIO(image_bytes)
                        image_file.name = os.path.basename(image_path)  # Just the filename with extension
                        image_files.append(image_file)
                
                # Call OpenAI API to generate a transition frame using GPT-Image-1
                try:
                    logger.info(f"Calling OpenAI API with GPT-Image-1 to generate transition frame")
                    
                    # GPT-Image-1 only supports a single image as input, so we use the first image
                    # and describe the second in the prompt
                    response = client.images.edit(
                        model="gpt-image-1",
                        image=image_files,  # Only use the first image as base
                        prompt=f"I want to generate a transition frame, with the first image on the left and the second image on the right, creating a camera-panning-right effect.",
                        size="1024x1024",  # We'll resize later if needed
                        quality="low",  # Using low quality to save costs
                    )
                    
                    # Check if response is valid
                    if not response.data or len(response.data) == 0:
                        raise Exception("No image data returned from OpenAI API")
                    
                    # Get the generated image data
                    image_url = response.data[0].url
                    if not image_url:
                        raise Exception("No image URL in response")
                    
                    logger.info(f"Successfully received response from OpenAI. Downloading image from URL.")
                    # Download the image from URL
                    image_response = requests.get(image_url)
                    if image_response.status_code != 200:
                        raise Exception(f"Failed to download image from URL: {image_url}")
                    
                    # Load the generated image
                    generated_img = Image.open(BytesIO(image_response.content))
                    
                    # Resize to match the original frame dimensions
                    if generated_img.size != (end_frame.shape[1], end_frame.shape[0]):
                        generated_img = generated_img.resize((end_frame.shape[1], end_frame.shape[0]))
                    
                    return np.array(generated_img)
                    
                except Exception as e:
                    logger.error(f"OpenAI Image Edit API error: {str(e)}")
                    # Fallback to blending
                    logger.info("Falling back to simple frame blending")
                    return (start_frame.astype(np.float32) * 0.5 + end_frame.astype(np.float32) * 0.5).astype(np.uint8)
            
            finally:
                # Clean up temporary files
                os.remove(start_file_path)
                os.remove(end_file_path)
                
        except ImportError:
            logger.error("OpenAI package not installed. Cannot generate transition frame.")
            # Return a simple blend of the two frames as a fallback
            return (start_frame.astype(np.float32) * 0.5 + end_frame.astype(np.float32) * 0.5).astype(np.uint8)
            
    def merge_videos(self, video_paths: List[str], output_path: str, mp4_crf: int = 10) -> str:
        """
        Merge multiple videos into one.
        
        Args:
            video_paths: List of paths to the videos to merge.
            output_path: Path to save the merged video.
            mp4_crf: CRF value for MP4 encoding (lower is better quality).
            
        Returns:
            Path to the merged video.
        """
        import tempfile
        import subprocess
        
        # Create a temporary file with the list of videos
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
            file_list = f.name
        
        try:
            # Use FFmpeg to concatenate the videos
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list,
                '-c:v', 'libx264',
                '-crf', str(mp4_crf),
                '-preset', 'slow',
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            return output_path
        
        finally:
            # Clean up the temporary file
            os.remove(file_list)