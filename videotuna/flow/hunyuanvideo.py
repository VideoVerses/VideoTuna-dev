import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union, Dict, Any

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist

from videotuna.flow.generation_base import GenerationFlow
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.hunyuan.hyvideo_i2v.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE, NEGATIVE_PROMPT_I2V
from videotuna.hunyuan.hyvideo_i2v.vae import load_vae
from videotuna.hunyuan.hyvideo_i2v.modules import load_model
from videotuna.hunyuan.hyvideo_i2v.text_encoder import TextEncoder
from videotuna.hunyuan.hyvideo_i2v.utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from videotuna.hunyuan.hyvideo_i2v.utils.lora_utils import load_lora_for_pipeline
from videotuna.hunyuan.hyvideo_i2v.modules.posemb_layers import get_nd_rotary_pos_embed
from videotuna.hunyuan.hyvideo_i2v.modules.fp8_optimization import convert_fp8_linear
from videotuna.hunyuan.hyvideo_i2v.diffusion.schedulers import FlowMatchDiscreteScheduler
from videotuna.hunyuan.hyvideo_i2v.diffusion.pipelines import HunyuanVideoPipeline
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from safetensors.torch import load_file
from omegaconf import ListConfig


###############################################
# 20250308 pftq: Riflex workaround to fix 192-frame-limit bug, credit to Kijai for finding it in ComfyUI and thu-ml for making it
# https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import numpy as np
from typing import Union,Optional
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis
    
class HunyuanVideoFlow(GenerationFlow):
    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        cond_stage_2_config: Dict[str, Any] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        *arggs, **kwargs
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
        self.args = args
        self.default_negative_prompt = NEGATIVE_PROMPT_I2V
        self.pipeline = HunyuanVideoPipeline(
            vae=self.first_stage_model,
            text_encoder=self.cond_stage_model,
            text_encoder_2=self.cond_stage_2_model,
            transformer=self.denoiser,
            scheduler=self.scheduler,
            progress_bar_config=None,
            args=args,
        )

        

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size

    

    # 20250317 pftq: Modified to use Riflex when >192 frames
    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W
    
        # Compute latent sizes based on VAE type
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]
    
        # Compute rope sizes
        if isinstance(self.denoiser.patch_size, int):
            assert all(s % self.denoiser.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.denoiser.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.denoiser.patch_size for s in latents_size]
        elif isinstance(self.denoiser.patch_size, list) or isinstance(self.denoiser.patch_size, ListConfig):
            assert all(
                s % self.denoiser.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.denoiser.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.denoiser.patch_size[idx] for idx, s in enumerate(latents_size)]
    
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # Pad time axis
    
        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly
    
        head_dim = self.denoiser.hidden_size // self.denoiser.heads_num
        rope_dim_list = self.denoiser.rope_dim_list or [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"
    
        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")
    
            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(size, device=self.device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]
    
            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def inference(
        self,
        config,
        prompt='一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光',
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        embedded_guidance_scale=6.0,
        batch_size=1,
        num_videos_per_prompt=1,
        i2v_mode=True,
        i2v_resolution="720p",
        i2v_image_path='0.jpg',
        i2v_condition_type=None,
        i2v_stability=True,
        ulysses_degree=1,
        ring_degree=1,
        xdit_adaptive_size=True,
        **kwargs,
    ):
        out_dict = dict()

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if guidance_scale == 1.0:
            negative_prompt = ""
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        img_latents = None
        semantic_images = None
        if i2v_mode:
            if i2v_resolution == "720p":
                bucket_hw_base_size = 960
            elif i2v_resolution == "540p":
                bucket_hw_base_size = 720
            elif i2v_resolution == "360p":
                bucket_hw_base_size = 480
            else:
                raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")

            semantic_images = [Image.open(i2v_image_path).convert('RGB')]
            origin_size = semantic_images[0].size

            crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
            aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
            closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

            if ulysses_degree != 1 or ring_degree != 1:
                closest_size = (height, width)
                resize_param = min(closest_size)
                center_crop_param = closest_size

                if xdit_adaptive_size:
                    original_h, original_w = origin_size[1], origin_size[0]
                    target_h, target_w = height, width

                    scale_w = target_w / original_w
                    scale_h = target_h / original_h
                    scale = max(scale_w, scale_h)

                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    resize_param = (new_h, new_w)
                    center_crop_param = (target_h, target_w)
            else:
                resize_param = min(closest_size)
                center_crop_param = closest_size

            ref_image_transform = transforms.Compose([
                transforms.Resize(resize_param),
                transforms.CenterCrop(center_crop_param),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
            semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img_latents = self.pipeline.vae.encode(semantic_image_pixel_values).latent_dist.mode()
                img_latents.mul_(self.pipeline.vae.config.scaling_factor)

            target_height, target_width = closest_size

        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        n_tokens = freqs_cos.shape[0]
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=torch.Generator(device=self.device),
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=False,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=img_latents,
            semantic_images=semantic_images,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
    
    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False) -> None:
        pass
