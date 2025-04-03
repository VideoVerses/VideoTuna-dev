import torch
import logging
import os
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys
import asyncio
from tqdm import tqdm

from videotuna.flow.generation_base import GenerationFlow
from videotuna.utils.common_utils import instantiate_from_config


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pickle
import torch, copy
from transformers.models.bert.modeling_bert import BertEmbeddings
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from videotuna.flow.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from videotuna.stepvideo.stepvideo.modules.model import StepVideoModel
from videotuna.stepvideo.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from videotuna.stepvideo.stepvideo.utils import VideoProcessor, with_empty_init
from videotuna.stepvideo.stepvideo.modules.model import RMSNorm
from videotuna.stepvideo.stepvideo.vae.vae import CausalConv, CausalConvAfterNorm, Upsample2D

mainlogger = logging.getLogger("mainlogger")
class StepVideoModelFlow(GenerationFlow):
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
        cond_stage_2_config: Dict[str, Any] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        save_path: str = './results',
        name_suffix: str = '',
        *args, **kwargs
    ):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            lr_scheduler_config=lr_scheduler_config,
            trainable_components=[]
        )

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.video_processor = VideoProcessor(save_path, name_suffix)
        
        self.scale_factor = 1.0
        self.torch_dtype = torch.bfloat16
        self.device_type = 'cuda'


    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.cond_stage_2_model.parameters())).dtype
        enable_vram_management(
            self.cond_stage_2_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                BertEmbeddings: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.cond_stage_model.parameters())).dtype
        enable_vram_management(
            self.cond_stage_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                RMSNorm: AutoWrappedModule,
                torch.nn.Embedding: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.denoiser.parameters())).dtype
        enable_vram_management(
            self.denoiser,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.first_stage_model.parameters())).dtype
        enable_vram_management(
            self.first_stage_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                CausalConv: AutoWrappedModule,
                CausalConvAfterNorm: AutoWrappedModule,
                Upsample2D: AutoWrappedModule
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        self.enable_cpu_offload()

    
    def encode_prompt(
        self,
        input_prompts: List[str],
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        prompts = [input_prompt + pos_magic for input_prompt in input_prompts]
        bs = len(prompts)
        prompts += [neg_magic] * bs
        
        prompt_embeds, prompt_embeds_mask = self.cond_stage_model(prompts)
        clip_embedding, _ = self.cond_stage_2_model(prompts)
        
        len_clip = clip_embedding.shape[1]
        prompt_embeds_mask = torch.nn.functional.pad(prompt_embeds_mask, (len_clip, 0), value=1)   ## pad attention_mask with clip's length 

        return prompt_embeds, clip_embedding, prompt_embeds_mask

    def check_inputs(self, num_frames, width, height):
        num_frames = max(num_frames//17*17, 1)
        width = max(width//16*16, 16)
        height = max(height//16*16, 16)
        return num_frames, width, height

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 64,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_frames, width, height = self.check_inputs(num_frames, width, height)
        shape = (
            batch_size,
            max(num_frames//17*3, 1),
            num_channels_latents,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )   # b,f,c,h,w
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if generator is None:
            generator = torch.Generator(device=self.device)

        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return latents



    @torch.no_grad()
    def inference(self, config, device='cuda'):
        neg_magic = config.neg_magic
        pos_magic = config.pos_magic
        batch_size = config.bs
        time_shift = config.time_shift
        num_inference_steps = config.num_inference_steps
        unconditional_guidance_scale = config.unconditional_guidance_scale
        do_classifier_free_guidance = unconditional_guidance_scale > 1.0
        

        prompt = self.load_inference_inputs(config.prompts, config.mode)
        # 3. Encode input prompt
        self.load_models_to_device(['cond_stage_model', 'cond_stage_2_model'])
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            input_prompts=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic
        )

        denoiser_dtype = self.denoiser.dtype
        prompt_embeds = prompt_embeds.to(denoiser_dtype).to(device)
        prompt_attention_mask = prompt_attention_mask.to(denoiser_dtype).to(device)
        prompt_embeds_2 = prompt_embeds_2.to(denoiser_dtype).to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift,
            device=device
        )

        # 5. Prepare latent variables
        num_channels_latents = self.denoiser.config.in_channels
        latents = self.prepare_latents(
            batch_size * config.n_samples_prompt,
            num_channels_latents,
            config.height,
            config.width,
            config.num_frames,
            torch.bfloat16,
            device,
            torch.Generator(device=device).manual_seed(config.seed)
        ).to(device)

        # 7. Denoising loop
        self.load_models_to_device(['denoiser'])
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(denoiser_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype).to(device)

                noise_pred = self.denoiser(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    return_dict=False,
                )
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                )
                
                progress_bar.update()

        if not torch.distributed.is_initialized() or int(torch.distributed.get_rank())==0:
            self.load_models_to_device(['cond_stage_model'])
            video = self.first_stage_model.decode(latents.to(denoiser_dtype).to(device) / self.scale_factor)
            video = self.video_processor.postprocess_video(video, output_file_name="test", output_type="mp4")
            return video
    

    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None):
        pass