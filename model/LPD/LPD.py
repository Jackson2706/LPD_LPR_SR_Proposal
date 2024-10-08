from abc import ABC, abstractmethod
from torch import nn

# Abstract class for License Plate Detection models
class LicensePlateDetection(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Abstract method for forward pass, must be implemented by subclasses"""
        pass