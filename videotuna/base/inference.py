import torch
import os
from einops import rearrange
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchvision
import torchvision.transforms as transforms


class InferenceBase:
    """
    Base class for inference models.
    Users should inherit from this class and override the necessary
    methods to define their training process.
    """

    def __init__(self):
        pass

    @staticmethod
    def process_savename(savename: List[str], n_per_prompt: int = 1, mode: str = 'default') -> List[str]:
        """
        Processes the save name to include the save path.

        :param savename: The name of the file to be saved.
        :param n_per_prompt: The number of samples per prompt. Default is 1.
        :param mode: The mode in which the save name is processed. Default is 'default'.
        :return: The processed save name.
        """
        if n_per_prompt == 1:
            if mode == 'default':
                newnames = [f"prompt-{idx+1:04d}" for idx in range(len(savename))]
            elif mode == 'prompt':
                newnames = []
                for idx, name in enumerate(savename):
                    name = name[:100]  # limit the length of the name
                    newname = f"{name}"
                    newnames.append(newname)
        elif n_per_prompt > 1:
            if mode == 'default':
                newnames = []
                for idx in range(len(savename)):
                    for i in range(n_per_prompt):
                        newnames.append(f"prompt-{idx+1:04d}-{i:02d}")
            elif mode == 'prompt':
                newnames = []
                for idx, name in enumerate(savename):
                    for i in range(n_per_prompt):
                        name = name[:100]
                        newnames.append(f"{name}-{i:02d}")
        else:
            raise ValueError("Invalid number of samples per prompt.")

        return newnames
    
    @staticmethod
    def save_video(
            vid_tensor: torch.Tensor,
            savepath: str,
            fps: int = 10
        ) -> None:
        """
        Save a video tensor to the specified path.

        :param vid_tensor: The video tensor to be saved.
        :param savepath: The path where the video will be saved.
        :param fps: Frames per second for the saved video. Default is 10.
        """
        # vid_tensor shape: [c, t, h, w]
        assert vid_tensor.dim() == 4, "Invalid video tensor shape."
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = rearrange(video, 'c t h w -> t c h w')
        video = (video + 1.0) / 2.0
        video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
        
        torchvision.io.write_video(
            savepath, video, fps=fps, video_codec="h264", options={"crf": "10"}
        )

    def save_videos(
            self,
            batch_tensors: torch.Tensor, 
            savedir: str, 
            filenames: List[str], 
            fps: int = 10
        ) -> None:
        """
        Save a batch of video tensors to the specified directory.

        :param batch_tensors: A tensor containing the batch of video data.
        :param savedir: The directory where the videos will be saved.
        :param filenames: A list of filenames for each video in the batch.
        :param fps: Frames per second for the saved videos. Default is 10.
        """
        # The batch shape is [bs, n_samples, c, t, h, w]
        bs = batch_tensors.shape[0]
        n_samples = batch_tensors.shape[1]
        assert batch_tensors.dim() == 6, "Invalid batch shape."
        assert n_samples * bs == len(filenames), "Number of filenames must match the batch size."

        c = 0
        for idx, vid_tensor in enumerate(batch_tensors):
            for i in range(n_samples):
                single_vid_tensor = vid_tensor[i]
                savepath = os.path.join(savedir, f"{filenames[c]}.mp4")
                self.save_video(single_vid_tensor, savepath, fps=fps)
                c += 1
    
    def save_videos_vbench(
            self, 
            batch_tensors: torch.Tensor, 
            savedir: str, 
            prompts: List[str], 
            format_file: dict, 
            fps: int = 10
        ) -> None:
        """
        Save a batch of video tensors to the specified directory with filenames based on prompts.

        :param batch_tensors: A tensor containing the batch of video data.
        :param savedir: The directory where the videos will be saved.
        :param prompts: A list of prompts used to generate filenames for each video.
        :param format_file: A dictionary to store the format of the file.
        :param fps: Frames per second for the saved videos. Default is 10.
        """
        # The batch shape is [bs, n_samples, c, t, h, w]
        b = batch_tensors.shape[0]
        n_samples = batch_tensors.shape[1]
        assert batch_tensors.dim() == 6, "Invalid batch shape."

        sub_savedir = os.path.join(savedir, "videos")
        os.makedirs(sub_savedir, exist_ok=True)

        for idx in range(b):
            prompt = prompts[idx]
            for n in range(n_samples):
                filename = f"{prompt}-{n}.mp4"
                format_file[filename] = prompt
                self.save_video(batch_tensors[idx, n], os.path.join(sub_savedir, filename), fps=fps)

    @staticmethod
    def load_prompts_from_txt(prompt_file: str) -> List[str]:
        """Load and return a list of prompts from a text file, stripping whitespace."""
        with open(prompt_file, "r") as f:
            lines = f.readlines()
        prompt_list = [line.strip() for line in lines if line.strip() != ""]
        return prompt_list
    
    def load_inference_inputs(self, prompts: Optional[Union[str, Path]], mode: str = 't2v'):
        """
        Loads the prompts and conditions for the conditional stage model.

        :param prompts: List of prompts to be loaded.
        :param mode: The mode in which the prompts are loaded. `t2v` or `i2v`.
        :return: `t2v` -> prompts; 
                 `i2v` -> prompts + images.
        """
        assert prompts is not None, "Please provide a valid prompts or prompts path."

        if mode == 't2v':
            # load inputs for t2v
            if os.path.isfile(prompts) and prompts.endswith('.txt'):
                prompt_list = self.load_prompts_from_txt(prompts)
            else:
                print("Process the input path as a prompt")
                prompt_list = [prompts]
            return prompt_list
        elif mode == 'i2v':
            # TODO: load images
            pass
        else:
            raise NotImplementedError("Invalid mode.")

    
    # TODO: Add more methods as needed
    # - sample
    # - save results
