"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
import glob
import os
import sys
import time
from typing import Literal

import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)

sys.path.insert(0, os.getcwd())
from diffusers.utils import export_to_video, load_image, load_video

from videotuna.utils.inference_utils import get_target_filelist, load_prompts_from_txt
from videotuna.utils.common_utils import monitor_resources, save_metrics


def generate_video(
    model_input: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal[
        "t2v", "i2v", "v2v"
    ],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    enable_sequential_cpu_offload: bool = False,
    enable_model_cpu_offload: bool = False,
    enable_vae_slicing: bool = False,
    enable_vae_tiling: bool = False
):
    """
    Generates a video based on the given input and saves it to the specified path.

    Parameters:
    - model_input (str): can be a string prompt or a path to a prompt file for t2v, or a directory containing images or videos for i2v and v2v.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path or directory where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """
    if not output_path.endswith(".mp4"):  # output_path is a directory
        os.makedirs(output_path, exist_ok=True)

    if model_input.endswith(".txt"):
        # model_input is a file for t2v
        prompts = load_prompts_from_txt(prompt_file=model_input)
        image_or_video_paths = [None] * len(prompts)
    elif os.path.isdir(model_input):
        if generate_type == "i2v":
            # model_input is a directory for i2v
            prompt_file = get_target_filelist(model_input, ext="txt")[0]
            prompts = load_prompts_from_txt(prompt_file=prompt_file)
            images = get_target_filelist(model_input, ext="png,jpg,webp,jpeg")
            image_or_video_paths = images
        elif generate_type == "v2v":
            # model_input is a directory for v2v
            prompt_file = get_target_filelist(model_input, ext="txt")[0]
            prompts = load_prompts_from_txt(prompt_file=prompt_file)
            videos = [
                os.path.join(model_input, f)
                for f in os.listdir(model_input)
                if f.endswith(".mp4")
            ]
            image_or_video_paths = videos
    else:
        assert isinstance(model_input, str)
        prompts = [model_input]
        image_or_video_paths = [None]

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="test_1",
        )
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if enable_vae_slicing:
        pipe.vae.enable_slicing()
    if enable_vae_tiling:
        pipe.vae.enable_tiling()

    start_time = time.time()
    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    gpu_metrics = []
    time_metrics = []
    for i, (prompt, image_or_video_path) in enumerate(
        zip(prompts, image_or_video_paths)
    ):
        output_path_ = (
            os.path.join(output_path, f"{i:03d}-{prompt}.mp4")
            if os.path.isdir(output_path)
            else output_path
        )
        result_with_metrics = inference(image_or_video_path, num_inference_steps, guidance_scale, num_videos_per_prompt, generate_type, seed, pipe, prompt)
        video_generate = result_with_metrics['result']
        gpu_metrics.append(result_with_metrics.get('gpu', -1.0))
        time_metrics.append(result_with_metrics.get('time', -1.0))
        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        export_to_video(video_generate, output_path_, fps=8)
    save_metrics(gpu=gpu_metrics, time=time_metrics, config=None, savedir=output_path)

    print(f"Total time taken: {time.time() - start_time:.2f}s")
    avg_time = (time.time() - start_time) / len(prompts) / num_videos_per_prompt
    print(f"Average time taken per prompt: {avg_time:.2f}s")

@monitor_resources(return_metrics=True)
def inference(image_or_video_path, num_inference_steps, guidance_scale, num_videos_per_prompt, generate_type, seed, pipe, prompt):
    if generate_type == "i2v":
        image = load_image(image=image_or_video_path)
        video_generate = pipe(
                prompt=prompt,
                image=image,  # The path of the image to be used as the background of the video
                num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
                num_inference_steps=num_inference_steps,  # Number of inference steps
                num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
                use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(
                    seed
                ),  # Set the seed for reproducibility
            ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
    else:
            # v2v
        video = load_video(image_or_video_path)
        video_generate = pipe(
                prompt=prompt,
                video=video,  # The path of the video to be used as the background of the video
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                # num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(
                    seed
                ),  # Set the seed for reproducibility
            ).frames[0]
        
    return video_generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--generate_type",
        type=str,
        default="t2v",
        help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')",
    )
    parser.add_argument(
        "--model_input",
        type=str,
        default="",
        help="The description of the video to be generated",
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="The path of the pre-trained model to be used",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The path of the LoRA weights to be used",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="The rank of the LoRA weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.mp4",
        help="The path where the generated video will be saved",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="The scale for classifier-free guidance",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of steps for the inference process",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="The data type for computation (e.g., 'float16' or 'bfloat16')",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for reproducibility"
    )
    parser.add_argument(
        "--enable_vae_tiling", action="store_true", help="enable vae tiling"
    )
    parser.add_argument(
        "--enable_vae_slicing", action="store_true", help="enable vae slicing"
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload", action="store_true", help="enable sequential cpu offload"
    )
    parser.add_argument(
        "--enable_model_cpu_offload", action="store_true", help="enable model cpu offload"
    )


    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        model_input=args.model_input,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload,
        enable_vae_slicing=args.enable_vae_slicing,
        enable_vae_tiling=args.enable_vae_tiling,
    )
