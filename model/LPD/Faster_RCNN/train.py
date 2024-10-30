import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .BackBone import FeatureExtractor
from config import config
from .dataset import LicensePlateDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1280, 720))
        ])

img_dir = "C:\\Users\\hieuh\\Downloads\\project\\CCPD\\dataset\\1_batch\\images"
label_dir = "C:\\Users\\hieuh\\Downloads\\project\\CCPD\\dataset\\1_batch\\labels"

train_data = LicensePlateDataset(img_dir, label_dir , transform= transform)
train_loader = DataLoader(train_data, 2)

def training_loop(model, learning_rate, train_dataloader, n_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_list = []

    for epoch in tqdm(range(n_epochs), desc='Training Epochs'):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_keypoints_batch ,gt_classes_batch in train_dataloader:
            img_batch = img_batch.to(device)
            gt_bboxes_batch = gt_bboxes_batch.to(device)
            gt_classes_batch = gt_classes_batch.to(device)

            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_list.append(total_loss)

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss:.4f}")

    return loss_list

if __name__ == '__main__':
    backbone = FeatureExtractor(backbone="vgg16", pretrained=True, freeze_backbone=True).to(device)
    faster_rcnn_model = config['model']['LPD'](img_size=(1280, 720), roi_size=(2, 2), feature_extractor=backbone).to(device)
    learning_rate = 0.0001
    n_epochs = 50
    loss_list = training_loop(faster_rcnn_model, learning_rate, train_loader, n_epochs)