import torch.nn as nn
import torchvision

from .utils import *

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=False):
        super(FeatureExtractor, self).__init__()
        self.backbone_name = backbone
        self.backbone = self.load_backbone(backbone, pretrained)
        if freeze_backbone:
            self.freeze_backbone()

    def load_backbone(self, backbone, pretrained):
        if backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            # Remove avgpool and FC layer
            return nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            return model.features  # Use only feature layers
        elif backbone == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            return model.features  # Use the feature extractor part
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x.to(device))
        return x