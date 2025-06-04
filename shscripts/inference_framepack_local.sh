# ckpt='checkpoints/wan/Wan2.1-I2V-14B-480P/'
# ckpt="./results/train/train_framepack_i2v_hunyuan_lora_20250511235315/checkpoints/only_trained_model"
config='configs/010_framepack/framepack_i2v_hunyuan_local.yaml'
prompt_dir="inputs/i2v/576x1024"
savedir="results/i2v/framepack"

python3 scripts/inference_new.py \
    --config "$config" \
    --prompt_dir "$prompt_dir" \
    --savedir "$savedir" \
    --height 240 \
    --width 416 \
    --frames 81 \
    --seed 44 \
    --num_inference_steps 40 \
    --time_shift 3.0 \
    --enable_model_cpu_offload \
    # --ckpt_path "$ckpt" \
        
