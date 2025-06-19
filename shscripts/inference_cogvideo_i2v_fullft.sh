config=configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml
ckpt=${YOUR_CKPT_PATH}
prompt_dir=inputs/i2v/576x1024

current_time=$(date +%Y%m%d%H%M%S)
savedir="results/inference/i2v/cogvideox-i2v-fullft-$current_time"

python3 scripts/inference_cogvideo.py \
    --config $config \
    --ckpt_path $ckpt \
    --prompt_dir $prompt_dir \
    --savedir $savedir \
    --bs 1 --height 480 --width 720 \
    --fps 16 \
    --seed 6666 \
    --mode i2v \
    --denoiser_precision bf16
