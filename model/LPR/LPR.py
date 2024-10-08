from abc import ABC
from torch import nn

class LicensePlateReader(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass