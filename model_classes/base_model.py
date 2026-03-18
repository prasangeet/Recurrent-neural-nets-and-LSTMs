import torch.nn as nn 

class BaseSequenceModel(nn.Module):
    """
    The Base model that will be used for All sequence Models

    This class defines the common interface for all models 
    Like RNN, BiLSTM, and the Attention RNN.
    """
    def __init__(self) -> None:
        """
        Initialization of the base model.
        """
        super().__init__()

    def forward(self, x, hidden=None):
        """
        Forward Pass of the model 

        This function returns the logits and the updated hidden state
        """
        raise NotImplementedError("Forward function must be implemented")
