ckpt='checkpoints/videocrafter/t2v_v2_512/'
config='configs/001_videocrafter2/vc2_t2v_320x512_refactor.yaml'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/test"

python3 scripts/inference_new.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savedir $savedir \
