import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        """For `Trainer` class."""
        super().__init__()

    def forward(self, input):
        raise NotImplementedError
