from config import config
from model import LicensePlateDataset

import torch
from PIL import Image
import matplotlib.pyplot as plt

class Output:
    def __init__(self, image, model):
        self.image = image
        self.model = model 

    def inference(self):
        self.image = self.model(self.image)
        self.image = self.image.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        return self.image
    
LR_folder_path = config["LR_folder_path"]
HR_folder_path = config["HR_folder_path"]
LR_sample_path = r"D:\DDPM-Pytorch\data64\train\20240413_175544_1.jpg"
model_weight_path = config["model_weight_path"]["UNET"]

dataset = LicensePlateDataset(LR_folder_path, HR_folder_path, (256,256))
LR_sample = Image.open(LR_sample_path)
LR_transform = dataset.transforms(LR_sample).unsqueeze(0).to("cuda")


model = config['model']['UNET'].to("cuda")
model.load_state_dict(torch.load(model_weight_path))

output = Output(LR_transform, model)
HR_output = output.inference()

plt.subplot(1,2,1)
plt.imshow(LR_transform.squeeze(0).permute(1,2,0).cpu().numpy())
plt.title("LOW RESOLUTION")
plt.subplot(1,2,2)
plt.imshow(HR_output)
plt.title("HIGH RESOLUTION")
plt.show()

