ckpt='checkpoints/hunyuanvideo/HunyuanVideo-I2V'
dit_weight='checkpoints/hunyuanvideo/HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt'
config='configs/007_hunyuanvideo/hunyuanvideo_i2v.yaml'
prompt_dir="inputs/i2v/576x1024"
savedir="results/i2v/hunyuan"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --dit_weight "$dit_weight" \
    --config "$config" \
    --prompt_dir "$prompt_dir" \
    --savedir "$savedir" \
    --height 720 \
    --width 1280 \
    --i2v_resolution "720p" \
    --frames 129 \
    --seed 44 \
    --num_inference_steps 50 
