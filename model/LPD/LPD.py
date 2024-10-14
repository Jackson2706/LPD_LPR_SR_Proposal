from abc import ABC, abstractmethod
from torch import nn

# Abstract class for License Plate Detection models
class LicensePlateDetection(ABC, nn.Module):
    def __init__(self, name) -> None:
        super(LicensePlateDetection, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, x):
        """Abstract method for forward pass, must be implemented by subclasses"""
        pass

    def getName(self):
        return self.name