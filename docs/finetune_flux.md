
# Introduction
This document provides instructions for fine-tuning the Flux.1-dev model.

# Preliminary steps
1. **Install the environment** (see [Installation]()). 
2. **Log in to Hugging Face to get the access to the pretrained Flux model.** The pretrained model will be automatically downloaded when lauch the training.   
    **(1) Log in in the Hugging face accoun**t from the model webpage [Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), to be granted access to this model. 
    - More Flux models can be check in the [Flux repo](https://github.com/black-forest-labs/flux?tab=readme-ov-file#models) 

    **(2) Run `huggingface-cli login` in your terminal to log in to your Hugging Face account**  
    - To log in, `huggingface_hub` requires a token generated from [this link](https://huggingface.co/settings/tokens). Get the token from Hugging Face and enter the token in your terminal.

# Steps of Simple Fine-tuning
1. Put images in `data/images/${DataName}`. We provide example images that can be manually downloaded at [this link](https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/nezha.zip), or download and unzip via
```
wget https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/nezha.zip
unzip nezha.zip -d data/images/nezha
```
3. Set the exp configs in the file `configs/006_flux/config.json` and `configs/006_flux/multidatabackend.json`
    <details>
      <summary>Click to view the introduction to these arguments</summary>

      
      **Necessary arguments that you need to modify to train different loras.** 

      config.json  
      - `output_dir`: the directory for saving trained lora models and intermediate results.  
      - `validation_prompt`: the testing prompt for validation during training. It should contain the concept name used in training labels.  

      multidatabackend.json  
      - `instance_data_dir`: the image directory. set to `data/images/${DataName}`
      - `caption`: the simple caption that be used for all images. 
      
      **Optional arguments that you may need to adjust to match more advanced requirements.**  
      config.json
      - `max_train_steps`: the total steps for training.  
      - `num_train_epochs`: Total number of training epochs (-1 means determined by steps).
      - `lora_rank`: the rank of the LoRA models, the bigger, the more learnable parameters.
      - `learning_rate`: controls how much the model weights are adjusted per update, balancing convergence speed and stability.
      - `checkpointing_steps`: the steps intersection for saving each LoRA checkpoint.
      - `checkpoints_total_limit`: the total number of saved model checkpoints.
      - `resume_from_checkpoint`: Resume training from the latest checkpoint.
      - `data_backend_config`: Path to the data backend configuration file.
      - `pretrained_model_name_or_path`: Name or path of the pre-trained model.
      - `seed`: Random seed for reproducibility.
      - `train_batch_size`: Batch size for training.
      - `gradient_checkpointing`: Whether to enable gradient checkpointing.
      - `disable_tf32`: Whether to disable TF32.
      - `mixed_precision`: Type of mixed precision.
      - `optimizer`: Type of optimizer.
      - `lr_warmup_steps`: Number of warmup steps for learning rate.
      - `lr_scheduler`: Type of learning rate scheduler.
      - `resolution_type`: Type of resolution.
      - `resolution`: Image resolution.
      - `validation_seed`: Random seed for validation.
      - `validation_steps`: Number of validation steps.
      - `validation_resolution`: Image resolution for validation.
      - `validation_guidance`: Guidance coefficient for validation.
      - `validation_guidance_rescale`: Guidance rescale for validation.
      - `validation_num_inference_steps`: Number of inference steps for validation.
      - `aspect_bucket_rounding`: Rounding precision for image aspect ratio bucketing.
      - `minimum_image_size`: Minimum image size.
      - `disable_benchmark`: Whether to disable benchmarking.
      - `lora_type`: Type of LoRA (Low-Rank Adaptation).
      - `model_type`: Type of the model.
      - `model_family`: Family of the model.
      - `write_batch_size`: Batch size for writing.
      - `caption_dropout_probability`: Probability of caption dropout.
    </details>


3. Run the commands in the terminal to launch training.
    ```
    poetry run train-flux-lora
    ```
4. After training, run the commands in the terminal to inference your personalized videotuna models.
    ```
    poetry run inference-flux-lora \
    --prompt "nezha is riding a bike" \
    --lora_path ${lora_path} \
    --out_path ${out_path}
    ```
    - ${out_path} should be a file path like `image.jpg`  

    You can also inference multiple prompts by passing a txt file:
    ```
    poetry run inference-flux-lora \
    --prompt data/prompts/nezha.txt \
    --lora_path ${lora_path} \
    --out_path ${out_path}
    ```
    - ${out_path} should be a directory.


