export TOKENIZERS_PARALLELISM=false

VC2CKPT="checkpoints/videocrafter/t2v_v2_512/"  # pretrained checkpoint of videocrafter2
CONFIG='configs/001_videocrafter2/vc2_t2v_320x512.yaml'             # experiment config: model+data+training
TRAINEDCKPT="results/train/20250223144241_test/checkpoints/only_trained_model/denoiser-000-000000010.ckpt"  # [Optional] trained checkpoint

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="results/train"                                             # root directory for saving multiple experiments
EXPNAME="test"                                     # experiment name 
CURRENT_TIME=$(date +%Y%m%d%H%M%S)                                  # current time

# run
python scripts/train_new.py \
-t \
--sdckpt $SDCKPT \
--ckpt $VC2CKPT \
--base $CONFIG \
--logdir $RESROOT \
--name ${CURRENT_TIME}_${EXPNAME} \
--devices '0,' \
--trained_ckpt $TRAINEDCKPT
