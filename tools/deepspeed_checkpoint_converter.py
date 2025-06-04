# Please refer to https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#saving-training-checkpoints
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import torch

# The dir contains your deepspeed checkpoints. The dir should contains "lattest" file. 
# One example file path is: results/train/xxxxxxx_hunyuanvideo_t2v_lora/checkpoints/epoch=161.ckpt
checkpoint_dir = "results/train/train_framepack_i2v_hunyuan_lora_20250511235315/checkpoints/flow/flow-047-000001950.ckpt"

# Path to save your converted checkpoint. The checkpoint can be directly loaded with "torch.load" function
save_path = "results/train/train_framepack_i2v_hunyuan_lora_20250511235315/checkpoints/only_trained_model/denoiser-047-000001950-converted.ckpt"


state_dict = get_fp32_state_dict_from_zero_checkpoint(
    checkpoint_dir
)

checkpoint = {"state_dict": state_dict}

torch.save(checkpoint, save_path)

print(f"Checkpoint saved to {save_path}")
