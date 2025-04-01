from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed

import torch
import os
from flask import Flask, Response, jsonify, request, Blueprint
from flask_restful import Api, Resource
import pickle
import argparse
import threading
import argparse


    
if __name__ == "__main__":
    args = parse_args()

    setup_seed(args.seed)
        
    vae_dir = os.path.join(args.model_dir, args.vae_dir)
    llm_dir = os.path.join(args.model_dir, args.llm_dir)
    clip_dir = os.path.join(args.model_dir, args.clip_dir)

    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16)
    pipeline.setup_dir(vae_dir, llm_dir, clip_dir)
    pipeline.enable_vram_management(num_persistent_param_in_dit=0)

    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
    #python videotuna/stepvideo/run_parallel.py --model_dir /project/llmsvgen/songsong/DiffSynth-Studio/models/stepfun-ai/stepvideo-t2v --vae_dir vae --llm_dir step_llm --clip_dir hunyuan_clip  --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 50  --cfg_scale 9.0 --time_shift 13.0