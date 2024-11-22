export TOKENIZERS_PARALLELISM=false

# exp settings
EXPNAME="004_cogvideox2b"                          # experiment name 
CONFIG='configs/004_cogvideox/cogvideo2b.yaml'       # experiment config 
RESROOT="results/cogvideo_train"                   # experiment saving directory

# run
current_time=$(date +%Y%m%d%H%M%S)
python scripts/train.py \
-t \
--name "$current_time"_$EXPNAME \
--base $CONFIG \
--logdir $RESROOT \
--devices '0,' \
lightning.trainer.num_nodes=1 \
--auto_resume False