# Installation
To use deepspeed Zero3 training, please review the following preparation steps.

```shell
poetry run install-deepspeed
```

# Usage
```shell
sh shscripts/train_hunyuanvideo_t2v_lora_deepspeed.sh
```
After training, one additional checkpoints converting step is needed. The script is in the:
```shell
tools/deepspeed_checkpoint_converter.py
```