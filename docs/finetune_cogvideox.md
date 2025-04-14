
# Introduction
- This document provides instructions for fine-tuning the CogvideoX model.
- It supports both text-to-video and image-to-video.

# Preliminary steps
1. Install the environment (see [Installation]()).
2. Download the CogvideoX checkpoints. 

# Steps of Simple Fine-tuning
**Finetune CogVideoX Text-to-Video:**
1. Download the example training videos manually from [this link](https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip), or download via `wget`:
    ```
    wget https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip
    cd data
    unzip apply_lipstick.zip -d apply_lipstick
    ```
    Make sure the data is putted at `data/apply_lipstick/metadata.csv`

2. Run the commands in the terminal to launch training.
    ```
    bash shscripts/train_cogvideox_t2v_lora.sh
    ```
3. After training, run the commands to inference your personalized models.
    ```
    bash shscripts/inference_cogvideo_t2v_lora.sh
    ```
    - You need to provide the checkpoint path to the `ckpt` argument in the above shell script.  

    Note: 
    - The training and inference use the default model config from `configs/004_cogvideox/cogvideo5b.yaml`


**Finetune CogVideoX Image-to-Video:**
1. Assue your data is the above example data, run the commands in the terminal to launch training.
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
