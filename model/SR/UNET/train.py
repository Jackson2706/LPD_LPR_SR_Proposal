from dataset import LicensePlateDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import UNET
import torch.nn as nn

import os

epochs = 10
device = 'cuda'
loss_fn = nn.MSELoss()

LR_path = r"D:\LicensePlate\project_enhance_quality_license_plate\data_no_augmentation\LR_data\LR_data"
HR_path = r"D:\LicensePlate\project_enhance_quality_license_plate\data_no_augmentation\HR_data\HR_data"

image_dataset = LicensePlateDataset(LR_path, HR_path, (256, 256))
dataloader = DataLoader(image_dataset, batch_size=16, shuffle=True)
model = UNET(in_channels=3, out_channels=3)
model.to(device)
optimizer = Adam(model.parameters(),lr=0.001)
                 
for epoch in range(epochs):
  model.train()
  sum_loss = 0
  n=len(dataloader)
  for LR_image, HR_image in dataloader:
        LR_image, HR_image = LR_image.to(device), HR_image.to(device)
        optimizer.zero_grad()
        HR_output = model(LR_image)
        loss = loss_fn(HR_output, LR_image)
        loss.backward()
        optimizer.step()
        sum_loss+=loss
  print("Epoch {} : Loss {}".format(epoch+1,sum_loss/n))