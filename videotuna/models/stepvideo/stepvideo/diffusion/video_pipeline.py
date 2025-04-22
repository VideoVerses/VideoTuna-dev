# Copyright 2025 StepFun Inc. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pickle
import torch, copy
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
import asyncio

from ..modules.model import StepVideoModel
from .scheduler import FlowMatchDiscreteScheduler
from ..utils import VideoProcessor, with_empty_init
import os

from transformers.models.bert.modeling_bert import BertEmbeddings
from ..modules.model import RMSNorm
from ..vae.vae import CausalConv, CausalConvAfterNorm, Upsample2D

def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)
    

class AutoWrappedLinear(torch.nn.Linear):

    @with_empty_init
    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        super().__init__(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)


def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, total_num_param=0):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(module, module_map, module_config, max_num_param, overflow_module_config, total_num_param)
    return total_num_param


def enable_vram_management(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0)
    model.vram_management_enabled = True



@dataclass
class StepVideoPipelineOutput(BaseOutput):
    video: Union[torch.Tensor, np.ndarray]
    
    
class StepVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using StepVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`StepVideoModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae_url:
            remote vae server's url.
        caption_url:
            remote caption (stepllm and clip) server's url.
    """

    def __init__(
        self,
        transformer: StepVideoModel,
        scheduler: FlowMatchDiscreteScheduler,
        vae_dir: str = '',
        caption_dir: tuple = ('', ''),
        save_path: str = './results',
        name_suffix: str = '',
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
        )
        

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.video_processor = VideoProcessor(save_path, name_suffix)
        self.model_names = ['vae', 'text_encoder', 'clip', 'transformer']

        self.torch_dtype = torch.bfloat16
        self.device_type = 'cuda'


        # self.vae_dir = vae_dir
        # self.llm_dir, self.clip_dir = caption_dir
        # self.setup_dir(self.vae_dir, self.llm_dir, self.clip_dir)
        
    def setup_dir(self, vae_dir, llm_dir, clip_dir, version=2):
        self.vae_dir = vae_dir
        self.llm_dir = llm_dir
        self.clip_dir = clip_dir
        self.text_encoder = self.build_llm(llm_dir)
        self.clip = self.build_clip(clip_dir)
        self.vae = self.build_vae(vae_dir, version)
        self.scale_factor = 1.0
        return self
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.clip.parameters())).dtype
        enable_vram_management(
            self.clip,
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
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
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
        dtype = next(iter(self.transformer.parameters())).dtype
        enable_vram_management(
            self.transformer,
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
                onload_device=self.device_type,
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
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
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

    def enable_cpu_offload(self):
        self.cpu_offload = True


    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()
    
    def build_llm(self, model_dir):
        from stepvideo.text_encoder.stepllm import STEP1TextEncoder
        text_encoder = STEP1TextEncoder(model_dir, max_length=320).eval()
        print("Inintialized text encoder...")
        return text_encoder
        
    def build_clip(self, model_dir):
        from stepvideo.text_encoder.clip import HunyuanClip
        clip = HunyuanClip(model_dir, max_length=77).eval()
        print("Inintialized clip encoder...")
        return clip

    def build_vae(self, vae_dir, version=2):
        from stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(vae_dir, model_name) 
        
        model = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=version,
        ).eval()
        print("Inintialized vae...")
        return model
    
    def encode_prompt(
        self,
        prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        device = self._execution_device
        prompts = [prompt+pos_magic]
        bs = len(prompts)
        prompts += [neg_magic]*bs
        
        data = self.embedding(prompts)
        prompt_embeds, prompt_attention_mask, clip_embedding = data['y'].to(device), data['y_mask'].to(device), data['clip_embedding'].to(device)

        return prompt_embeds, clip_embedding, prompt_attention_mask

    def embedding(self, prompts, *args, **kwargs):
        with torch.no_grad():
            try:
                y, y_mask = self.text_encoder(prompts)
                    
                clip_embedding, _ = self.clip(prompts)
                
                len_clip = clip_embedding.shape[1]
                y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)   ## pad attention_mask with clip's length 

                data = {
                    'y': y.detach().cpu(),
                    'y_mask': y_mask.detach().cpu(),
                    'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                }

                return data
            except Exception as err:
                print(f"{err}")
                return None

    def decode_vae(self, samples):
        samples = self.decode(samples)
        return samples
    
    def decode(self, samples, *args, **kwargs):
        with torch.no_grad():
            try:
                dtype = self.dtype
                device = self.device_type
                samples = self.vae.decode(samples.to(dtype).to(device) / self.scale_factor)
                if hasattr(samples,'sample'):
                    samples = samples.sample
                return samples
            except:
                torch.cuda.empty_cache()
                return None

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
            generator = torch.Generator(device=self._execution_device)

        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return latents


    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        time_shift: float = 13.0,
        neg_magic: str = "",
        pos_magic: str = "",
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "mp4",
        output_file_name: Optional[str] = "",
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `544`):
                The height in pixels of the generated image.
            width (`int`, defaults to `992`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `204`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `9.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. 
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            output_file_name(`str`, *optional*`):
                The output mp4 file name.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`StepVideoPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~StepVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`StepVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        self.load_models_to_device(['text_encoder', 'clip'])
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype).to(self.device_type)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype).to(self.device_type)
        prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype).to(self.device_type)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift,
            device=device
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.bfloat16,
            device,
            generator,
            latents,
        ).to(self.device_type)

        # 7. Denoising loop
        self.load_models_to_device(['transformer'])
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype).to(self.device_type)

                noise_pred = self.transformer(
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
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                )
                
                progress_bar.update()

        if not torch.distributed.is_initialized() or int(torch.distributed.get_rank())==0:
            if not output_type == "latent":
                self.load_models_to_device(['vae'])
                video = self.decode_vae(latents)
                video = self.video_processor.postprocess_video(video, output_file_name=output_file_name, output_type=output_type)
            else:
                video = latents

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (video, )

            return StepVideoPipelineOutput(video=video)
        

        