export TOKENIZERS_PARALLELISM=false

# dependencies
CONFIG="configs/007_hunyuanvideo/hunyuanvideo_diffuser_deepspeed.yaml"   # experiment config 

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="results/train"             # experiment saving directory
EXPNAME="hunyuanvideo_t2v_lora"          # experiment name 
CURRENT_TIME=$(date +%Y%m%d%H%M%S)  # current time

python scripts/train_deepspeed.py \
-t \
--base $CONFIG \
--logdir $RESROOT \
--name "$CURRENT_TIME"_$EXPNAME \
--devices '0,1,2,3' \
lightning.trainer.num_nodes=1 \
--auto_resume