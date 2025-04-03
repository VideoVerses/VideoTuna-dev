import torch
from videotuna.stepvideo.stepvideo.config import parse_args
import os


accepted_version = {
    '2.2': 'liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so',
    '2.3': 'liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so',
    '2.5': 'liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so',
}

try:
    #args = parse_args()
    version = '.'.join(torch.__version__.split('.')[:2])
    if version in accepted_version:
        torch.ops.load_library(os.path.join("/project/llmsvgen/songsong/DiffSynth-Studio/models/stepfun-ai/stepvideo-t2v", f'lib/{accepted_version[version]}'))
    else:
        raise ValueError("Not supported torch version for liboptimus")
except Exception as err:
    print(err)
