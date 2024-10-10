from abc import ABC
from torch import nn


# Abstract class for Super Resolution models
class SuperResolutionModel(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()


    # Abstract method for the forward pass, must be implemented by any subclass
    def forward(self, x):
        pass

