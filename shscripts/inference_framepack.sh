# ckpt='checkpoints/wan/Wan2.1-I2V-14B-480P/'
ckpt="./hf_download/"
config='configs/010_framepack/framepack_i2v_hunyuan.yaml'
prompt_dir="inputs/i2v/576x1024"
savedir="results/i2v/framepack"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --config "$config" \
    --prompt_dir "$prompt_dir" \
    --savedir "$savedir" \
    --seed 44 \
    --num_inference_steps 25
        
