export TOKENIZERS_PARALLELISM=false

CKPT="./hf_download"
CONFIG='configs/010_framepack/framepack_i2v_hunyuan_lora.yaml'        

RESROOT="results/train"                                             
EXPNAME="train_framepack_i2v_hunyuan_lora"                                     
CURRENT_TIME=$(date +%Y%m%d%H%M%S)                                  


python scripts/train_new.py -t \
--ckpt $CKPT \
--base $CONFIG \
--logdir $RESROOT \
--name "$EXPNAME"_"$CURRENT_TIME" \
--devices 0,1 \
--auto_resume