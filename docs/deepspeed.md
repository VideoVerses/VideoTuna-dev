# Installation
To use deepspeed Zero3 training, please review the following preparation steps.

1. Re-install deepspeed.

```shell
DS_BUILD_CPU_ADAM=1  BUILD_UTILS=1  pip install deepspeed -U
```

2. Install cuda-toolkit.

```shell
conda install -c "nvidia/label/cuda-xx.x.0" cuda-toolkit
```

# Usage
```shell
sh shscripts/train_hunyuanvideo_t2v_lora_deepspeed.sh
```
After training, one additional checkpoints converting step is needed. The script is in the:
```shell
tools/deepspeed_checkpoint_converter.py
```