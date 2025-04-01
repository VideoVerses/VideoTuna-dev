import math
import torch
import inspect
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3

# from videotuna.base.ddpm3d import DDPM
import pytorch_lightning as pl
from videotuna.utils.common_utils import instantiate_from_config
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    get_peft_model,
)
import numpy as np


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u



def prepare_sigmas(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    sigmas: torch.Tensor,
    batch_size: int,
    num_train_timesteps: int,
    flow_weighting_scheme: str = "none",
    flow_logit_mean: float = 0.0,
    flow_logit_std: float = 1.0,
    flow_mode_scale: float = 1.29,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        weights = compute_density_for_timestep_sampling(
            weighting_scheme=flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=flow_logit_mean,
            logit_std=flow_logit_std,
            mode_scale=flow_mode_scale,
            device=device,
            generator=generator,
        )
        indices = (weights * num_train_timesteps).long()
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")

    # print('device:', sigmas.device, 'indices:', indices.device)
    return sigmas[indices]


def expand_tensor_dims(tensor, ndim):
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def prepare_loss_weights(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    alphas: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    flow_weighting_scheme: str = "none",
) -> torch.Tensor:
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return compute_loss_weighting_for_sd3(sigmas=sigmas, weighting_scheme=flow_weighting_scheme)
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")


def prepare_target(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    noise: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        target = noise - latents
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        target = latents
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")

    return target


class HunyuanVideoWorkFlow(pl.LightningModule):
    def __init__(
        self,
        first_stage_config: Dict[str, Any],  # vae
        cond_stage_config: Dict[str, Any],  # text encoder
        cond_stage_config_2: Dict[str, Any],  # text encoder 2
        tokenizer_config: Dict[str, Any],  # tokenizer
        tokenizer_config_2: Dict[str, Any],  # tokenizer 2
        denoiser_config: Dict[str, Any],  # transformer
        scheduler_config: Dict[str, Any],  # scheduler
        learning_rate: float = 6e-6,
        adapter_config=None,
        deepspeed_config=None,
        logdir=None,
    ):
        super().__init__()
        self.model = instantiate_from_config(denoiser_config)

        self.logdir = logdir
        self.learning_rate = learning_rate
        # condtion stage use T5 class, which is availale at lvdm.module.encoders.condtion.FrozenT5Embedder
        # but we need to be aware of the model name and tokenizer name
        # here is the same with DDPM
        # print(first_stage_config)
        self.instantiate_first_stage(first_stage_config)
        # max_sequence_length=226
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_cond_stage_2(cond_stage_config_2)

        self.tokenizer = instantiate_from_config(tokenizer_config)
        self.tokenizer_2 = instantiate_from_config(tokenizer_config_2)

        # self.vae_scale_factor_spatial = (
        #     2 ** (len(self.vae.config.block_out_channels) - 1)
        #     if hasattr(self, "first_stage_model") and self is not None
        #     else 8
        # )
        # self.vae_scale_factor_temporal = (
        #     self.vae.config.temporal_compression_ratio
        #     if hasattr(self, "first_stage_model") and self.vae is not None
        #     else 4
        # )

        # self.video_processor = VideoProcessor(
        #     vae_scale_factor=self.vae_scale_factor_spatial
        # )
        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio 
            if getattr(self, "vae", None) 
            else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio 
            if getattr(self, "vae", None) 
            else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        self.model = instantiate_from_config(denoiser_config)
        # self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.scheduler = instantiate_from_config(scheduler_config)
        # print('self.device:', self.device)
        # exit()
        self.scheduler_sigmas = self.scheduler.sigmas.clone().to(self.device)
        # add adapter config (Support Lora and HRA )
        self.lora_args = []
        if adapter_config is not None:
            self.inject_adapter(adapter_config)
        if deepspeed_config is not None:
            self.deepspeed_config = deepspeed_config.params
        
    
    def inject_adapter(self, adapter_config):
        self.model.requires_grad_(False)
        self.model.enable_gradient_checkpointing()
        # print("Injecting lora adapter")
        transformer_adapter_config = instantiate_from_config(adapter_config)   
        # print(transformer_adapter_config)
        self.model = get_peft_model(self.model, transformer_adapter_config)
        self.model.print_trainable_parameters()
    
    ## VAE is named as first_stage_model 
    ## followed functions are all first stage related. 
    def instantiate_first_stage(self, config):
        # import pdb;pdb.set_trace()
        model = instantiate_from_config(config)
        self.vae = model.eval()
        # self.vae.train = disabled_train
        self.vae.requires_grad_(False)
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist

    def _decode_core(self, z, **kwargs):
        z = 1. / self.scale_factor * z

        if self.encoder_type == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE(z)
        results = self.vae.decode(z, **kwargs)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self._decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        """same as decode_first_stage but without decorator"""
        return self._decode_core(z, **kwargs)
    
    ## second stage : text condition and other condtions 
    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        # # in finetune cogvideox don't train as default
        self.cond_stage_model = model.eval()
        self.cond_stage_model.requires_grad_(False)
    
    def instantiate_cond_stage_2(self, config):
        model = instantiate_from_config(config)
        # # in finetune cogvideox don't train as default
        self.cond_stage_model_2 = model.eval()
        self.cond_stage_model_2.requires_grad_(False)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames
    

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_template=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_template is not None:
            if not isinstance(prompt_template, dict):
                raise ValueError(f"`prompt_template` has to be of type `dict` but is {type(prompt_template)}")
            if "template" not in prompt_template:
                raise ValueError(
                    f"`prompt_template` has to contain a key `template` but only found {prompt_template.keys()}"
                )
    

    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # device = device or self._execution_device
        # dtype = dtype or self.text_encoder.dtype

        device = self.device
        # TODO: fix data type 
        # dtype = torch.float32
        dtype = torch.float16

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2
        
        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        # todo: check if this is correct
        prompt_embeds = self.cond_stage_model(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt)

        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        # device = device or self._execution_device
        # dtype = dtype or self.text_encoder_2.dtype

        device = self.device
        # TODO: fix data type 
        # dtype = torch.float32
        dtype = torch.float16

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            # logger.warning( 
            #     "The following part of your input was truncated because CLIP can only handle sequences up to"
            #     f" {max_sequence_length} tokens: {removed_text}"
            # )

        prompt_embeds = self.cond_stage_model_2(text_input_ids.to(device), output_hidden_states=False).pooler_output
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds


    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None and pooled_prompt_embeds is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()
      

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def sample(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
    ) -> Union[Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. Note that the only available HunyuanVideo model is
                CFG-distilled, which means that traditional guidance between unconditional and conditional latent is
                not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        # 3. Encode input prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.model.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.model.config.in_channels
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            # torch.float32,
            torch.float16,
            device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.interrupt = False 
        self.model.cuda()

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()
        
        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            # video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        video = video[None,...]
        video = video.cpu()
        torch.cuda.empty_cache()
        return video 
    
    # training specific functions 
    def configure_optimizers(self):
        if self.deepspeed_config is not None and self.deepspeed_config.use_cpu_adam:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam([p for p in self.model.parameters() if p.requires_grad ], lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad ], lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        new_satate_dict = checkpoint["state_dict"]
        new_satate_dict = {k: v for k, v in new_satate_dict.items() if "lora" in k}
        checkpoint["state_dict"] = new_satate_dict
        return checkpoint
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass
    
    def encode_video(self,video):
        # print(f"video shape in encode_video: {video.shape}") # [61, 3, 544, 960]
        video = video.to(self.device, dtype=self.vae.dtype).unsqueeze(0) # [1, 61, 3, 544, 960]
        # video = video.to(self.device, dtype=self.vae.dtype).unsqueeze(0) # [1, 61, 3, 544, 960]
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W], [1, 3, 61, 544, 960]
        print(self.vae.dtype)
        print(self.dtype)
        
        latent_dist = self.vae.encode(video).latent_dist
        return latent_dist
   
    def get_batch_input(self, batch):
        """
        Prepare model batch inputs 
        """
        # equal to collate_fn
        # the resonable video latents range is [-5,5], approximately.
        # videos
        videos = [self.encode_video(video) for video in batch["video"]]
        videos = [video.sample() * self.vae.config.scaling_factor for video in videos]
        videos = torch.cat(videos, dim=0)
        videos = videos.to(memory_format=torch.contiguous_format).float()
        # prompt
        prompts = [item for item in batch["prompt"]]
        return {
            "videos": videos,
            "prompts": prompts,
        }
    
    def training_step(self, batch, batch_idx):
        batch = self.get_batch_input(batch)
        # model_input = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=self.vae.dtype)  # [B, F, C, H, W]
        model_input = batch["videos"].to(dtype=self.vae.dtype)
        prompts = batch["prompts"]
        # print(f"model_input.dtype: {model_input.dtype}")
        # print(f"self.vae.dtype: {self.vae.dtype}")
        # import pdb
        # pdb.set_trace()
        
        max_sequence_length = 256 # TODO: check this value
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                prompt_attention_mask=None,
                device=self.device,
                max_sequence_length=max_sequence_length,
            )
        # print(f"prompt_embeds.dtype: {prompt_embeds.dtype}")
        # print(f"pooled_prompt_embeds.dtype: {pooled_prompt_embeds.dtype}")
        # print(f"prompt_attention_mask.dtype: {prompt_attention_mask.dtype}")

        batch_size, num_frames, num_channels, height, width = model_input.shape

        # flow_weighting_scheme: str = "none"
        # flow_logit_mean: float = 0.0
        # flow_logit_std: float = 1.0
        # flow_mode_scale: float = 1.29
        
        sigmas = prepare_sigmas(
            scheduler=self.scheduler,
            sigmas=self.scheduler_sigmas.to(self.device),
            batch_size=batch_size,
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            flow_weighting_scheme="none",
            flow_logit_mean=0.0,
            flow_logit_std=1.0,
            flow_mode_scale=1.29,
            device=self.device,
            generator=None, # TODO: do we need to set the generator here?
        )

        timesteps = (sigmas * 1000.0).long()

        noise = torch.randn(
            model_input.shape,
            generator=None, # TODO: do we need to set the generator here?
            device=self.device,
            dtype=self.vae.dtype,
        )
        sigmas = expand_tensor_dims(sigmas, ndim=noise.ndim)

        noisy_latents = (1.0 - sigmas) * model_input + sigmas * noise
        noisy_latents = noisy_latents.to(model_input.dtype)
        # print(f"noisy_latents.dtype: {noisy_latents.dtype}")
        weights = prepare_loss_weights(
            scheduler=self.scheduler,
            alphas=None, # None for flow matching
            sigmas=sigmas,
            flow_weighting_scheme="none",
        )
        weights = expand_tensor_dims(weights, noise.ndim)

        # pred = self.model_config["forward_pass"](
        #     transformer=self.transformer,
        #     scheduler=self.scheduler,
        #     timesteps=timesteps,
        #     **latent_conditions,
        #     **text_conditions,
        # )
        guidance_scale = 1.0
        guidance = torch.tensor([guidance_scale] * noisy_latents.shape[0], dtype=noisy_latents.dtype, device=noisy_latents.device) * 1000.0
        # print(f"noisy_latents.dtype: {noisy_latents.dtype}")
        # print(f"timesteps.dtype: {timesteps.dtype}")
        # print(f"prompt_embeds.dtype: {prompt_embeds.dtype}")
        # print(f"pooled_prompt_embeds.dtype: {pooled_prompt_embeds.dtype}")
        
        model_output = self.model(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds.to(noisy_latents),
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds.to(noisy_latents),
                    guidance=guidance,
                    # attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
        # print(f"model_output.dtype: {model_output.dtype}")
        target = prepare_target(
            scheduler=self.scheduler, noise=noise, latents=model_input
        )
        # print(f"target.dtype: {target.dtype}")
        loss = weights.float() * (model_output.float() - target.float()).pow(2)
        # Average loss across all but batch dimension
        loss = loss.mean(list(range(1, loss.ndim)))
        # Average loss across batch dimension
        loss = loss.mean()
        return loss


    def training_step_old(self, batch, batch_idx):
        batch = self.get_batch_input(batch)
        # model_input = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=self.vae.dtype)  # [B, F, C, H, W]
        model_input = batch["videos"].to(dtype=self.vae.dtype)
        prompts = batch["prompts"]
        
        max_sequence_length = 256 # TODO: check this value
        with torch.no_grad():
            # prompt_embeds = self.encode_prompt(
            #     prompts,
            #     do_classifier_free_guidance=False,# set to false for train
            #     num_videos_per_prompt=1,
            #     max_sequence_length=max_sequence_length,
            #     device=self.device,
            #     dtype=self.vae.dtype,
            # )
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                prompt_attention_mask=None,
                device=self.device,
                max_sequence_length=max_sequence_length,
            )
            
        batch_size, num_frames, num_channels, height, width = model_input.shape
        
        # generate noise 
        # 
        
        # Sample noise that will be added to the latents
        noise = torch.randn_like(model_input)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

        # print('noisy_model_input.dtype', noisy_model_input.dtype)
        # print('timesteps.dtype', timesteps.dtype, timesteps.shape)
        # print('prompt_embeds.dtype', prompt_embeds.dtype)
        # print('prompt_attention_mask.dtype', prompt_attention_mask.dtype)
        # print('pooled_prompt_embeds.dtype', pooled_prompt_embeds.dtype)
        # prin
        # guidance = torch.tensor([self._guidance_scale], device=self.device, dtype=self.vae.dtype) * 1000.0
        guidance_scale = 1.0
        guidance = torch.tensor([guidance_scale] * noisy_model_input.shape[0], dtype=noisy_model_input.dtype, device=noisy_model_input.device) * 1000.0
        # print('guidance.dtype', guidance.dtype, guidance.shape)
        model_output = self.model(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    # attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
        model_pred = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = model_input
        # TODO: inherent loss computation from base class. 
        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        return loss


