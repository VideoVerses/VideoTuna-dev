# ckpt='checkpoints/wan/Wan2.1-I2V-14B-480P/'
ckpt="./hf_download/"
config='configs/010_framepack/framepack_i2v_hunyuan.yaml'
prompt_dir="/project/llmsvgen/yazhou/check-base/maintain/temp/VideoTuna-dev/test-i2v-480"
savedir="results/i2v/framepack"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --config "$config" \
    --prompt_dir "$prompt_dir" \
    --savedir "$savedir" \
    --height 480 \
    --width 832 \
    --frames 81 \
    --seed 44 \
    --num_inference_steps 40 \
    --time_shift 3.0 \
    --enable_model_cpu_offload
        
