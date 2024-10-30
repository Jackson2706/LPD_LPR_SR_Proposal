import torch.nn as nn
from .utils import *

class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, feature_map, proposals_list, gt_classes=None):

        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'

        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))

        cls_scores = self.cls_head(out)

        if mode == 'eval':
            return cls_scores
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())

        return cls_loss