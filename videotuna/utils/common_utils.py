import importlib
import os
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf
import time
import psutil
import subprocess
import sys
from functools import wraps
from loguru import logger

import cv2
import numpy as np
import torch
import torch.distributed as dist


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    istarget = False
    for para in para_list:
        if para in name:
            return True
    return istarget

def get_dtype_from_str(dtype_str):
    import torch
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float32)  # 默认返回float32

def get_params(config, resolve=True):
    params = config.get("params")
    if params is None:
        return dict()

    if resolve and isinstance(params, DictConfig):
        return OmegaConf.to_container(params, resolve=True)
    return params

# resolve will make params dict type rather than DictConfig type
def instantiate_from_config(config, resolve=True):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if "diffusers" in config["target"] or config["target"].startswith("transformers") or config.get("use_from_pretrained", False):
        return get_obj_from_str(config["target"]).from_pretrained(
            **get_params(config, resolve)
        )
    return get_obj_from_str(config["target"])(**get_params(config, resolve))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [
        np.load(os.path.join(data_dir, data_name))["arr_0"]
        for data_name in os.listdir(data_dir)
    ]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)["arr_0"] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group("nccl", init_method="env://")


def print_green(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def print_red(text):
    print(Fore.RED + text + Style.RESET_ALL)

def print_yellow(text):
    print(Fore.YELLOW + text + Style.RESET_ALL)


def monitor_resources(return_metrics=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            start_time = time.time()
            start_cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024 # GB

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024 # GB

            time_used = end_time - start_time
            cpu_mem_used = end_cpu_mem - start_cpu_mem

            logger.info(f"Time used: {time_used:.2f} seconds")
            logger.info(f"CPU memory change: {cpu_mem_used:.2f} GB")
            gpu_mem_used = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024 # GB
                logger.info(f"Peak GPU memory used: {gpu_mem_used:.2f} GB")

            if return_metrics:
                return {
                    "time_used_sec": round(time_used, 2),
                    "cpu_mem_used_gb": round(cpu_mem_used, 2),
                    "gpu_mem_used_gb": round(gpu_mem_used, 2) if gpu_mem_used is not None else None,
                    "result": result,
                }
            else:
                return result

        return wrapper
    return decorator

