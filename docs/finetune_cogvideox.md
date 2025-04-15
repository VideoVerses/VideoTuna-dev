
# Introduction
- This document provides instructions for fine-tuning the CogvideoX model.
- It supports both text-to-video and image-to-video.

# Preliminary steps
1. Install the videotuna environment (see [Installation](https://github.com/VideoVerses/VideoTuna?tab=readme-ov-file#1prepare-environment)).
2. Download the CogvideoX checkpoints (see [docs/checkpoints](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md)). 
3. Download the example training data.
You can download manually from [this link](https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip), or download via `wget`:
    ```
    wget https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip
    cd data
    unzip apply_lipstick.zip -d apply_lipstick
    ```
    Make sure the data is putted at `data/apply_lipstick/metadata.csv`

# Steps of Simple Fine-tuning
**Lora Fine-tuning of CogVideoX Text-to-Video:**

1. Run the commands in the terminal to launch training.
    ```
    bash shscripts/train_cogvideox_t2v_lora.sh
    ```
2. After training, run the commands to inference your personalized models.
    ```
    bash shscripts/inference_cogvideo_t2v_lora.sh
    ```
    - You need to provide the checkpoint path to the `ckpt` argument in the above shell script.  

    Note: 
    - The training and inference use the default model config from `configs/004_cogvideox/cogvideo5b.yaml`


**Lora Fine-tuning of CogVideoX Image-to-Video:**
1. Run the commands in the terminal to launch training.
    ```
    bash shscripts/train_cogvideox_i2v_lora.sh
    ```
2. After training, run the commands to inference your personalized models.
    ```
    bash shscripts/inference_cogvideo_i2v_lora.sh
    ```
    - You need to provide the checkpoint path to the `ckpt` argument in the above shell script.  

    Note: 
    - The training and inference use the default model config from `configs/004_cogvideox/cogvideo5b-i2v.yaml`

**Full Fine-tuning of CogVideoX Text-to-Video:**
1. Run the commands in the terminal to launch training.
    ```
    bash shscripts/train_cogvideox_t2v_fullft.sh
    ```
    We tested on 4 H800 GPUs. The training requires 68GB GPU memory.
2. After training, run the commands to inference your personalized models.
    ```
    shscripts/inference_cogvideo_t2v_fullft.sh
    ```
    - You need to provide the checkpoint path to the `ckpt` argument in the above shell script. Because the full fine-tuning uses deepspeed to reduce GPU memory, so the checkpoint is like `${exp_save_dir}/checkpoints/trainstep_checkpoints/epoch=xxxxxx-step=xxxxxxxxx.ckpt/checkpoint/mp_rank_00_model_states.pt`

    Note: 
    - The training and inference use the default model config from `configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml`

**Full Fine-tuning of CogVideoX Image-to-Video:**

Same as above full fine-tuning of text-to-video. 
1. Training:
```
bash shscripts/train_cogvideox_i2v_fullft.sh
```
2. Inference:
```
shscripts/inference_cogvideo_i2v_fullft.sh
```