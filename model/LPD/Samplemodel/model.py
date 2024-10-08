from ..LPD import LicensePlateDetection
from torch import nn

# Subclass implementing the LicensePlateDetection abstract class
class MyLicensePlateModel(LicensePlateDetection):
    def __init__(self):
        super().__init__()
        # Convolutional layer with 3 input channels and 16 output channels
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer for final classification or detection
        self.fc = nn.Linear(16 * 32 * 32, 10)

    # Forward pass implementation
    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply ReLU activation function
        x = nn.functional.relu(x)
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        # Apply the fully connected layer
        x = self.fc(x)
        return x