
# Introduction
- This document provides instructions for fine-tuning the VideoCrafter2 model.
- It supports both full fine-tuning and lora fine-tuning.



# Preliminary steps
  1) [Install the environment](#1prepare-environment)
  2) [Prepare the dataset   ](#41-prepare-dataset) to get the example dataset
```
$ ll Dataset/ToyDataset/

ToyDataset/
    ├── toydataset.csv
    ├── videos/
        ├── video1.mp4
        ├── video2.mp4
        ...
```
  3) [Download the checkpoints](docs/CHECKPOINTS.md) and get the checkpoint
```
  $ ll checkpoints/videocrafter/t2v_v2_512/model.ckpt
```
Then, run this command to convert the VC2 checkpoint as we make minor modifications on the keys of the state dict of the checkpoint. The converted checkpoint will be automatically save at `checkpoints/videocrafter/t2v_v2_512/model_converted.ckpt`.
```
python tools/convert_checkpoint.py \
--input_path checkpoints/videocrafter/t2v_v2_512/model.ckpt
```
Then you will get the following checkpoints
```
  $ ll checkpoints/videocrafter/t2v_v2_512_split
  cond_stage.ckpt
  denoiser.ckpt
  first_stage.ckpt
  model_new.ckpt
```

# Steps of Simple Fine-tuning
**1. Full Fine-tuning of VideoCrafter2 Text-to-Video:**

**(1) Train:** Run this command to start training on the single GPU. 
```
bash shscripts/train_videocrafter_v2.sh
```
or
```
poetry run train-videocrafter-v2
```

The training results will be automatically saved at `results/train/${CURRENT_TIME}_${EXPNAME}`. The checkpoints will be save every 100 iteractions.

**(2) Inference:** Replace denoiser.ckpt with the newly trained denoiser.ckpt saved in above directory (e.g., `results/train/${CURRENT_TIME}_${EXPNAME}/checkpoints/only_trained_model`) and perform inference via running:
```
bash shscripts/inference_vc2_t2v_320x512.sh
```

**2. Lora Fine-tuning of VideoCrafter2 Text-to-Video:**  
**(1) Train:**

```
bash shscripts/train_videocrafter_lora.sh
```
or
```
poetry run train-videocrafter-v2
```

- The training and inference use the default model config from `configs/001_videocrafter2/vc2_t2v_lora.yaml`

**(2) Inference:**
```
bash shscripts/inference_vc2_t2v_320x512_lora.sh
```
