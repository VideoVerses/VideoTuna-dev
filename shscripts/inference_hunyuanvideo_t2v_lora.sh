# ----------------------diffusers based pl inference ----------------------
# ‘configs/004_cogvideox/cogvideo2b.yaml’ or 'configs/004_cogvideox/cogvideo5b.yaml'
config='configs/007_hunyuanvideo/hunyuanvideo_diffuser.yaml'
prompt_file="inputs/t2v/hunyuanvideo/tyler_swift_video/labels.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-hunyuanvideo"
# ckpt="{YOUR_CKPT_PATH}"
ckpt="results/train/20250228203955_hunyuanvideo_t2v_lora/checkpoints/epoch=430.ckpt"

python3 scripts/inference_cogvideo.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $savedir \
--bs 1 --height 256 --width 256 \
--fps 16 \
--seed 6666 \
