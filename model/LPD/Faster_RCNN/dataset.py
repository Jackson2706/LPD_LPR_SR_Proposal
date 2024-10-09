import os
import torch
from PIL import Image

from dataset.dataset import CustomImageDataset

class LicensePlateDataset(CustomImageDataset):
    '''
    A Pytorch Dataset class to load images, bounding boxes, and keypoints lazily.
    Returns:
    images: torch.Tensor of size (C, H, W)
    gt_bboxes: torch.Tensor of size (max_objects, 4)
    gt_keypoints: torch.Tensor of size (max_objects, 8) # 8 for 4 keypoints (x1, y1, x2, y2, x3, y3, x4, y4)
    gt_classes: torch.Tensor of size (max_objects)
    '''
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
        self.label_paths = sorted([os.path.join(label_dir, label) for label in os.listdir(label_dir)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image lazily
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img)

        label_path = self.label_paths[idx]
        gt_boxes, gt_keypoints, gt_classes = self.parse_annotation(label_path)

        return img_tensor, gt_boxes, gt_keypoints, gt_classes

    def parse_annotation(self, label_path):
        gt_boxes_all = []
        gt_keypoints_all = []
        gt_classes_all = []
        
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line_split = line.strip().split()
            # Extract keypoints
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line_split[1:])
            
            # Bounding box calculation
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)
            bbox = torch.Tensor([x_min, y_min, x_max, y_max])

            # Keypoints as a flat array of 8 values (x1, y1, x2, y2, x3, y3, x4, y4)
            keypoints = torch.Tensor([x1, y1, x2, y2, x3, y3, x4, y4])

            gt_boxes_all.append(bbox)
            gt_keypoints_all.append(keypoints)
            gt_classes_all.append(0)  # Single class for license plates

        gt_boxes = torch.stack(gt_boxes_all)
        gt_keypoints = torch.stack(gt_keypoints_all)
        gt_classes = torch.tensor(gt_classes_all)

        return gt_boxes, gt_keypoints, gt_classes
