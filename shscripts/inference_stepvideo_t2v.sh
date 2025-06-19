ckpt='checkpoints/stepvideo/stepvideo-t2v/'
config='configs/009_stepvideo/stepvideo_t2v.yaml'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/t2v/stepvideo"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --config "$config" \
    --prompt_file "$prompt_file" \
    --savedir "$savedir" \
    --height 544 \
    --width 992 \
    --frames 51 \
    --seed 44 \
    --num_inference_steps 50 \
    --enable_model_cpu_offload
