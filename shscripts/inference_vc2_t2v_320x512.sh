ckpt='checkpoints/videocrafter/t2v_v2_512_refactor/'
config='configs/001_videocrafter2/vc2_t2v_320x512.yaml'
prompt_file="inputs/t2v/prompts.txt"

python3 scripts/inference_new.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savefps 30
