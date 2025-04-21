
# Prepare checkpoints

This document contains commands for preparing model checkpoints and the final checkpoint organization structure.


### 1. Supported Models

|T2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|HunyuanVideo|720x1280x129|[Hugging Face](https://huggingface.co/tencent/HunyuanVideo)
|Mochi|848x480, 3s|[Hugging Face](https://huggingface.co/genmo/mochi-1-preview)
|CogVideoX-2B|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b)
|CogVideoX-5B|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)
|Open-Sora 1.0|512×512x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)
|VideoCrafter2|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-5B-I2V|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
|DynamiCrafter|576x1024x16|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|

* Note: H: height; W: width; L: length


### 2. Download checkpoints
Please run the following commands in your terminal to download the checkpoints for each model.
```
mkdir checkpoints

# ---------------------------- T2V ----------------------------

# ---- CogVideo (diffusers) ----
mkdir -p checkpoints/cogvideo; cd checkpoints/cogvideo
git clone https://huggingface.co/THUDM/CogVideoX-2b         # This are checkpoints for CogVideoX T2V-2B
git clone https://huggingface.co/THUDM/CogVideoX-5b         # This are checkpoints for CogVideoX T2V-5B
git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V     # This are checkpoints for CogVideoX I2V-5B
git clone https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT  # This are checkpoints for CogVideoX 1.5-5B (both T2V and I2V)

# ---- HunyuanVideo (diffusers) ----
cd VideoTuna   # Make sure you are under the root path of VideoTuna
mkdir checkpoints/hunyuan
cd checkpoints/hunyuan
git lfs install
git clone https://huggingface.co/hunyuanvideo-community/HunyuanVideo


# ---- Open-Sora ----
mkdir -p checkpoints/open-sora/t2v_v10
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
#
mkdir -p checkpoints/open-sora/t2v_v11
cd checkpoints/open-sora/t2v_v11
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3
cd ../../..
#
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3


# ---- Videocrafter ----
mkdir checkpoints/videocrafter/

mkdir checkpoints/videocrafter/t2v_v2_512
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v2_512  # videocrafter2-t2v-512
python tools/videocrafter_checkpoint_converter.py

mkdir checkpoints/videocrafter/t2v_v1_1024
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v1_1024 # videocrafter1-t2v-1024


# ---- StepVideo ----
mkdir checkpoints/stepvideo/
cd checkpoints/stepvideo
huggingface-cli download stepfun-ai/stepvideo-t2v --local-dir ./stepvideo-t2v
cd ../..

# ---- Wan ----
mkdir checkpoints/wan/
cd checkpoints/wan
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
cd ../..


# ---- HunyuanVideo ----
mkdir -p checkpoints/hunyuanvideo/
huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./checkpoints/hunyuanvideo/HunyuanVideo-I2V
cd checkpoints/hunyuanvideo/HunyuanVideo-I2V
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./text_encoder_i2v
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
cd ../..


# ---------------------------- I2V ----------------------------
# ---- Dynamicrafter ----
mkdir checkpoints/dynamicrafter/
mkdir checkpoints/dynamicrafter/i2v_576x1024

wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt -P checkpoints/dynamicrafter/i2v_576x1024  # dynamicrafter-i2v-1024

# ---- Videocrafter ----
mkdir -p checkpoints/videocrafter/i2v_v1_512

wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt -P checkpoints/videocrafter/i2v_v1_512 # videocrafter1-i2v-512

# ---- Stable Diffusion checkpoint for VC2 Training ----
mkdir -p checkpoints/stablediffusion/v2-1_512-ema
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -P checkpoints/stablediffusion/v2-1_512-ema

# ---- Wan ----
mkdir -p checkpoints/wan/
cd checkpoints/wan
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
cd ../..

# ---------------------------- V2V ----------------------------
# ---- ModelScope Video-to-Video ----
cd checkpoints
# please ensure that you have installed lfs. If not, you can install it by running the following command:
git lfs install
# after installing lfs, you can clone the Video-to-Video checkpoints
git clone https://www.modelscope.cn/iic/Video-to-Video.git

```


### 3. Checkpoint Orgnization Structure
After downloading, the model checkpoints should be placed as follows:

```
VideoTuna/
    └── checkpoints/
        ├── cogvideo/
        │   └── CogVideoX-2b/
        │   └── CogVideoX-5b/
        │   └── CogVideoX-5b-I2V/
        ├── hunyuanvideo/
        │   ├── HunyuanVideo-I2V/
        │   │   └── hunyuan-video-i2v-720p/
        │   │   └── text_encoder_2
        │   │   └── text_encoder_i2v
        │   └── HunyuanVideo/
        │       └── hunyuan-video-t2v-720p/
        │       └── llava-llama-3-8b-v1_1-transformers/
        │       └── text_encoder
        │       └── text_encoder_2
        ├── dynamicrafter/
        │   └── i2v_576x1024/
        │       └── model.ckpt
        ├── videocrafter/
        │   ├── t2v_v2_512/
        │   │   └── model.ckpt
        │   ├── t2v_v2_512_split/
        │   │   └── cond_stage.ckpt
        │   │   └── denoiser.ckpt
        │   │   └── first_stage.ckpt
        │   │   └── model_new.ckpt
        │   ├── t2v_v1_1024/
        │   │   └── model.ckpt
        │   └── i2v_v1_512/
        │       └── model.ckpt
        └── open-sora/
            ├── t2v_v10/
            │   ├── OpenSora-v1-16x256x256.pth
            │   └── OpenSora-v1-HQ-16x512x512.pth
            ├── t2v_v11/
            │   ├── OpenSora-STDiT-v2-stage2/
            │   └── OpenSora-STDiT-v2-stage3/
            └── t2v_v12/
                ├── OpenSora-STDiT-v3/
                └── OpenSora-VAE-v1.2/
```

If you do not follow these locations, please modify the default checkpoint path argument during training/inference.
