from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from colorama import Fore, Style

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from videotuna.base.train import TrainBase
from videotuna.base.inference import InferenceBase
from videotuna.utils.common_utils import instantiate_from_config, print_green, print_yellow



class GenerationFlow(TrainBase, InferenceBase):
    """
    The GenerationFlow class is a generative model class that inherits from both TrainBase and InferenceBase.
    It manages the instantiation of different stages of a generative process, including a denoiser and a scheduler.
    It also configures optimizers and learning rate schedulers for training.

    The main components of the model are:
        - `first_stage`: a VAE model that encodes the input video into a latent space and decodes it back to the original video.
        - `cond_stage`: a conditional model that takes the latent space and the conditioning text as input and generates the output video.
        - `denoiser`: a denoiser model that takes the noisy output of the `cond_stage` and tries to remove the noise.
        - `scheduler`: a scheduler that controls denosing and sampling.
    """

    def __init__(self,
                 first_stage_config: Dict[str, Any],
                 cond_stage_config: Dict[str, Any],
                 denoiser_config: Dict[str, Any],
                 scheduler_config: Dict[str, Any],
                 cond_stage_2_config: Dict[str, Any] = None,
                 lr_scheduler_config: Dict[str, Any] = None,
                 trainable_components: Union[str, List[str]] = "denoiser",
                 ):
        """
        Initializes the GenerationFlow class with configurations for different stages and components.

        :param first_stage_config: Dictionary containing configuration for the first stage model.
        :param cond_stage_config: Dictionary containing configuration for the conditional stage model.
        :param cond_stage_2_config: Dictionary containing configuration for the conditional stage model 2, can be none.
        :param denoiser_config: Dictionary containing configuration for the denoiser model.
        :param scheduler_config: Dictionary containing configuration for the learning rate scheduler.
        :param lr_scheduler_config: Dictionary containing configuration for the learning rate scheduler.
        :param trainable_components: The components of the model that should be trainable.
        """
        super().__init__()

        # instantiate the modules
        self.components = []
        logger.info("creating denoiser")
        self.instantiate_denoiser(denoiser_config)

        logger.info("creating first stage")
        self.instantiate_first_stage(first_stage_config)
        logger.info("creating cond stage")
        self.instantiate_cond_stage(cond_stage_config)
        logger.info("creating cond stage 2")
        self.instantiate_cond_stage_2(cond_stage_2_config)

        logger.info("creating scheduler")
        self.scheduler = instantiate_from_config(scheduler_config)
        self.components.append('scheduler')

        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.cond_stage_2_config = cond_stage_2_config
        self.denoiser_config = denoiser_config
        self.scheduler_config = scheduler_config

        # this is learning rate scheduler
        self.use_lr_scheduler = lr_scheduler_config is not None
        if self.use_lr_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        
        # set trainable components
        self.set_trainable_components(trainable_components)
        

    def instantiate_first_stage(self, config: Dict[str, Any]):
        """
        Instantiates the first stage model of the generative process.

        :param config: Dictionary containing configuration for the first stage model.
        """
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        self.components.append('first_stage_model')
    
    def instantiate_cond_stage(self, config: Dict[str, Any]):
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional stage model.
        """
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        self.components.append('cond_stage_model')
        
    def instantiate_cond_stage_2(self, config: Dict[str, Any]):
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional stage model.
        """
        self.cond_stage_2_model = None
        if config is not None:
            model = instantiate_from_config(config)
            self.cond_stage_2_model = model.eval()
            for param in self.cond_stage_2_model.parameters():
                param.requires_grad = False
            self.components.append('cond_stage_2_model')
    
    def instantiate_denoiser(self, config: Dict[str, Any]):
        """
        Instantiates the denoiser model of the generative process.

        :param config: Dictionary containing configuration for the denoiser model.
        """
        model = instantiate_from_config(config)
        self.denoiser = model.eval()
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.components.append('denoiser')

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the generative model.

        :return: A list containing the optimizer and optionally a list containing the learning rate scheduler.
        """

        lr = self.learning_rate
        params = list(self.model.parameters())
        logger.info(f"@Training [{len(params)}] Full Paramters.")

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)
        ## lr scheduler
        if self.use_lr_scheduler:
            logger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_lr_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer

    def configure_lr_schedulers(self, optimizer):
        """
        Configures the learning rate scheduler based on the provided configuration.

        :param optimizer: The optimizer for which the scheduler is being configured.
        :return: A dictionary containing the scheduler, interval, and frequency.
        """
        assert 'target' in self.lr_scheduler_config
        scheduler_name = self.lr_scheduler_config.target.split('.')[-1]
        interval = self.lr_scheduler_config.interval
        frequency = self.lr_scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.lr_scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': interval,
                            'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.lr_scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                            'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                            'interval': interval,
                            'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler
    
    def set_trainable_components(
        self,
        components: Union[str, List[str]] = "denoiser",
    ):
        """
        Sets the components of the generative model that should be trainable.

        :param components: The components to be set as trainable.
        """
        if isinstance(components, str):
            components = [components]

        for component in components:
            model = getattr(self, component)
            if model is None:
                raise ValueError(f"Invalid component name: {component}")
            model.train()
            for param in self.first_stage_model.parameters():
                param.requires_grad = True
                
        print_green(f"Set the following components as trainable: {components}")
    
    def load_trained_ckpt(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        trained_model_name: Optional[str] = "denoiser",
    ):
        """
        Loads the weights of the specific trained model from a checkpoint file.
        """
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        # check if the component is in the flow
        ckpt_name = ckpt_path.split('/')[-1].split('-')[0]
        if ckpt_name in self.components:
            trained_model_name = ckpt_name

        if trained_model_name == "first_stage":
            self.first_stage_model = self.load_model(self.first_stage_model, ckpt_path)
        elif trained_model_name == "cond_stage":
            self.cond_stage_model = self.load_model(self.cond_stage_model, ckpt_path)
        elif trained_model_name == "denoiser":
            self.denoiser = self.load_model(self.denoiser, ckpt_path)
        else:
            raise ValueError(f"Invalid trained model name: {trained_model_name}")

        print_green(f"Successfully loaded a trained model {trained_model_name} from checkpoint.")
    
    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False) -> None:
        """
        Loads the weights of the model from a checkpoint file.

        :param ckpt_path: Path to the checkpoint file.
        :param ignore_missing_ckpts: If True, ignores missing checkpoints.
        """
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        ckpt_path = Path(ckpt_path)
        # load first_stage_model
        if (ckpt_path / "first_stage.ckpt").exists():
            self.first_stage_model = self.load_model(self.first_stage_model, ckpt_path / "first_stage.ckpt")
            print_green("Successfully loaded first_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of first_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of fisrt stage model file not found.")

        # load cond_stage_model
        if (ckpt_path / "cond_stage.ckpt").exists():
            self.cond_stage_model = self.load_model(self.cond_stage_model, ckpt_path / "cond_stage.ckpt")
            print_green("Successfully loaded cond_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of cond_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of cond_stage model file not found.")
        
        # load denoiser
        if (ckpt_path / "denoiser.ckpt").exists():
            self.denoiser = self.load_model(self.denoiser, ckpt_path / "denoiser.ckpt")
            print_green("Successfully loaded denoiser from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of denoiser file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of denoiser file not found.")
    

    def enable_vram_management(self):
        logger.info("enable_vram_management: default moving to cuda")
        self.cuda()
    

    def enable_cpu_offload(self):
        self.cpu_offload = True


    def load_models_to_device(self, loadmodel_names=[]):
        skip_components = ['scheduler']
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            logger.info("cpu offload is closed, skipping")
            return
        # offload the unneeded models to cpu
        for model_name in self.components:
            if model_name in skip_components:
                logger.info(f"{model_name} no need cpu offload, skipping")
                continue

            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        logger.info(f"{model_name} cpu offloading using offload method")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        logger.info(f"{model_name} cpu offloading using to cpu method")
                        model.cpu()

        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    logger.info(f"{model_name} onloading using onload method")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    logger.info(f"{model_name} onloading using to device method")
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()
    
    @staticmethod
    def load_model(model: nn.Module, ckpt_path: Optional[Union[str, Path]] = None):
        """
        Loads the weights of the model from a checkpoint file.

        :param model: The model to be loaded.
        :param ckpt_path: Path to the checkpoint file.
        """
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict)
            return model
        else:
            raise FileNotFoundError("Checkpoint of model file not found.")