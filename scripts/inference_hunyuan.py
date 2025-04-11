import os
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, os.getcwd())
sys.path.insert(1, f"{os.getcwd()}/src")
from videotuna.models.hyvideo.config import parse_args
from videotuna.models.hyvideo.inference import HunyuanVideoSampler
from videotuna.models.hyvideo.utils.file_utils import save_videos_grid
from videotuna.utils.inference_utils import load_prompts_from_txt
from videotuna.utils.common_utils import monitor_resources, save_metrics

def main():
    args = parse_args()
    print(args)
    if args.prompt.endswith(".txt"):
        prompts = load_prompts_from_txt(prompt_file=args.prompt)
    else:
        prompts = [args.prompt]

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = (
        args.save_path
        if args.save_path_suffix == ""
        else f"{args.save_path}_{args.save_path_suffix}"
    )
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args
    )

    # Get the updated args
    args = hunyuan_video_sampler.args
    gpu_metrics = []
    time_metrics = []

    # Start sampling
    for prompt in prompts:
        result_with_metrics = inference(args, prompt, hunyuan_video_sampler)
        outputs = result_with_metrics['result']
        samples = outputs["samples"]
        gpu_metrics.append(result_with_metrics.get('gpu', -1.0))
        time_metrics.append(result_with_metrics.get('time', -1.0))

        # Save samples
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            video_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
            save_videos_grid(sample, video_save_path, fps=24)
            logger.info(f"Sample save to: {video_save_path}")
    save_metrics(gpu=gpu_metrics, time=time_metrics, config=args, savedir=save_path)

@monitor_resources(return_metrics=True)
def inference(args, prompt, hunyuan_video_sampler):
    outputs = hunyuan_video_sampler.predict(
        prompt=prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    
    return outputs


if __name__ == "__main__":
    main()
