import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm, trange

sys.path.insert(0, os.getcwd())
sys.path.insert(1, f"{os.getcwd()}/src")
# from videotuna.base.ddim import DDIMSampler
from videotuna.scheduler import DDIMSampler
from videotuna.base.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from videotuna.utils.args_utils import prepare_args
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.inference_utils import (
    load_image_batch,
    load_inputs_i2v,
    load_model_checkpoint,
    load_prompts_from_txt,
    sample_batch_i2v,
    sample_batch_t2v,
    save_videos,
    save_videos_vbench,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="inference mode: t2v/i2v",
    )
    #
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="model config (yaml) path")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="a text file containing many prompts for text-to-video",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="a input dir containing images and prompts for image-to-video/interpolation",
    )
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument(
        "--standard_vbench",
        action="store_true",
        default=False,
        help="inference standard vbench prompts",
    )
    #
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    #
    parser.add_argument(
        "--height", type=int, default=None, help="video height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="video width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="video frame number, in pixel space"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="video motion speed. 512 or 1024 model: large value -> slow motion; 256 model: large value -> large motion;",
    )
    parser.add_argument(
        "--n_samples_prompt",
        type=int,
        default=None,
        help="num of samples per prompt",
    )
    #
    parser.add_argument("--bs", type=int, default=None, help="batch size for inference")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument(
        "--uncond_prompt",
        type=str,
        default="",
        help="unconditional prompts, or negative prompts",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=None,
        help="prompt classifier-free guidance",
    )
    parser.add_argument(
        "--unconditional_guidance_scale_temporal",
        type=float,
        default=None,
        help="temporal consistency guidance",
    )
    # dc args
    parser.add_argument(
        "--multiple_cond_cfg",
        action="store_true",
        default=False,
        help="i2v: use multi-condition cfg or not",
    )
    parser.add_argument(
        "--cfg_img",
        type=float,
        default=None,
        help="guidance scale for image conditioning",
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="generate looping videos or not",
    )
    parser.add_argument(
        "--gfi",
        action="store_true",
        default=False,
        help="generate generative frame interpolation (gfi) or not",
    )
    # lora args
    parser.add_argument(
        "--lorackpt",
        type=str,
        default=None,
        help="[Optional] checkpoint path for lora model. ",
    )
    #
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    return parser


def load_inputs(args):
    """
    load inputs:
        t2v: prompts
        i2v: prompts + images
    """
    assert (
        args.prompt_file is not None or args.prompt_dir is not None
    ), "Error: input file/dir NOT Found!"

    if args.prompt_file is not None:
        assert os.path.exists(args.prompt_file)
        # load inputs for t2v
        prompt_list = load_prompts_from_txt(args.prompt_file)
        num_prompts = len(prompt_list)
        filename_list = [f"prompt-{idx+1:04d}" for idx in range(num_prompts)]
        image_list = None
    elif args.prompt_dir is not None:
        assert os.path.exists(args.prompt_dir)
        # load inputs for i2v
        filename_list, image_list, prompt_list = load_inputs_i2v(
            args.prompt_dir,
            video_size=(args.height, args.width),
            video_frames=args.frames,
        )
    return prompt_list, image_list, filename_list


def run_inference(args, gpu_num=1, rank=0, **kwargs):
    """
    Inference t2v/i2v models
    """
    # load and prepare config
    assert Path(args.config).exists(), f"Error: config file {args.config} NOT Found!"
    config = OmegaConf.load(args.config)
    config = prepare_args(args, config)
    inference_config = config.pop("inference", OmegaConf.create())

    seed_everything(inference_config.seed)

    # create flow
    flow_config = config.pop("flow", OmegaConf.create())
    flow = instantiate_from_config(flow_config)
    flow.from_pretrained(inference_config.ckpt_path)
    flow = flow.cuda()
    flow.eval()

    # flow inference
    flow.inference(inference_config)


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_inference(args)
