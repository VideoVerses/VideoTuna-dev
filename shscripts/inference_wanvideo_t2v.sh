#480P
# ckpt='checkpoints/wan/Wan2.1-T2V-14B/'
# config='configs/008_wanvideo/wan2_1_t2v_14B_refactor.yaml'
# prompt_file="inputs/t2v/prompts.txt"
# savedir="results/t2v/wanvideo/480P"

# python3 scripts/inference_new.py \
#     --ckpt_path "$ckpt" \
#     --config "$config" \
#     --prompt_file "$prompt_file" \
#     --savedir "$savedir" \
#     --height 480 \
#     --width 832 \
#     --frames 81 \
#     --seed 44 \
#     --time_shift 3.0 \
#     --num_inference_steps 50 \

#720P
ckpt='checkpoints/wan/Wan2.1-T2V-14B/'
config='configs/008_wanvideo/wan2_1_t2v_14B_refactor.yaml'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/t2v/wanvideo/720P"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --config "$config" \
    --prompt_file "$prompt_file" \
    --savedir "$savedir" \
    --height 720 \
    --width 1280 \
    --frames 81 \
    --seed 44 \
    --time_shift 5.0 \
    --num_inference_steps 50 \