import torch
from typing import Any, Dict, List, Optional, Union


class InferenceBase:
    """
    Base class for inference models.
    Users should inherit from this class and override the necessary
    methods to define their training process.
    """

    def __init__(self, device: str = "cpu"):
        # Initialize the device used for inference, default is set to "cpu"
        self.device = device
    
    @torch.no_grad()
    def predict(self, model: Any, inputs: Union[List[Any], Any], **kwargs) -> Any:
        """
        This method should be overridden by subclasses to implement the actual prediction logic
        Using @torch.no_grad() to disable gradient calculation, which reduces memory consumption and speeds up computations during inference
        Args:
          model (Any): The model to use for prediction
          inputs (Union[List[Any], Any]): The input data for the model, can be a single input or a list of inputs
          **kwargs: Additional keyword arguments that can be passed to the predict method
        Returns:
          Any: The prediction result, type can vary depending on the model and inputs
        """
        raise NotImplementedError
    
    # TODO: Add more methods as needed
    # - process the input promts (text, or string)
    # - sample
    # - save results
