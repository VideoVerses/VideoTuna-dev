import os
import torch
from collections import OrderedDict
ckpt = torch.load("checkpoints/videocrafter/t2v_v2_512/model.ckpt")
state_dict = ckpt['state_dict']

denoiser_ckpt = {'state_dict': OrderedDict()}
first_stage_ckpt = {'state_dict': OrderedDict()}
cond_stage_ckpt = {'state_dict': OrderedDict()}
new_ckpt = {'state_dict': OrderedDict()}
for k, v in state_dict.items():
    if 'model.diffusion_model' in k:
        key_list = k.split('.')
        new_list = key_list[2:]
        new_key = '.'.join(new_list)
        denoiser_ckpt['state_dict'][new_key] = v
        print(f'{new_key} saved to denoiser_ckpt')
    elif 'first_stage_model' in k:
        key_list = k.split('.')
        new_list = key_list[1:]
        new_key = '.'.join(new_list)
        first_stage_ckpt['state_dict'][new_key] = v
        print(f'{new_key} saved to first_stage_ckpt')
    elif 'cond_stage_model' in k:
        key_list = k.split('.')
        new_list = key_list[1:]
        new_key = '.'.join(new_list)
        cond_stage_ckpt['state_dict'][new_key] = v
        print(f'{new_key} saved to cond_stage_ckpt')
    else:
        new_ckpt['state_dict'][k] = v

os.makedirs("checkpoints/videocrafter/t2v_v2_512_split", exist_ok=True)
torch.save(new_ckpt, "checkpoints/videocrafter/t2v_v2_512_split/model_new.ckpt")
torch.save(denoiser_ckpt, "checkpoints/videocrafter/t2v_v2_512_split/denoiser.ckpt")
torch.save(first_stage_ckpt, "checkpoints/videocrafter/t2v_v2_512_split/first_stage.ckpt")
torch.save(cond_stage_ckpt, "checkpoints/videocrafter/t2v_v2_512_split/cond_stage.ckpt")

print('Finish!')