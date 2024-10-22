import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
# sys.path.append(os.path.abspath('./model/SR'))
# import SR
from ..SR import SuperResolutionModel

in_channels = 3
out_channels = 3

class DoubleConv2d(SuperResolutionModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        x=self.conv(x)
        return x
    
class UNET(SuperResolutionModel):
    def __init__(self, in_channels=in_channels, out_channels=out_channels, features=[64,128]):
        super().__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.Maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels,feature))
            in_channels=feature
            
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2,out_channels=feature,padding=2,stride=2,kernel_size=2)
            )
            self.ups.append(DoubleConv2d(feature*2,feature))

        self.bottom=DoubleConv2d(features[-1],features[-1]*2)
        self.lastConv2d=nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections=[]
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.Maxpool2d(x)

        x=self.bottom(x)
        skip_connections=skip_connections[::-1]
        
        for i in range(0,len(self.ups),2):
            x=self.ups[i](x)
            skip_connection=skip_connections[i//2]
            
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x=torch.cat((x,skip_connection),dim=1)
            x=self.ups[i+1](x)
        
        x=self.lastConv2d(x)
        return x