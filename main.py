from model.SR.UNET.model import UNET
from model.SR.UNET.dataset import LicensePlateDataset
import matplotlib.pyplot as plt
import torch
from PIL import Image


class Output:
    def __init__(self, image, model):
        self.image = image
        self.model = model 

    def inference(self):
        self.image = self.model(self.image)
        self.image = self.image.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        return self.image
    
LR_path = r"D:\LicensePlate\project_enhance_quality_license_plate\data_augmentation\img_HRsumary4357\img_HRsumary"
HR_path = r"D:\LicensePlate\project_enhance_quality_license_plate\data_augmentation\img_LRsumary4357\img_LRsumary"
LR_sample_path = r"D:\LicensePlate\project_enhance_quality_license_plate\data_augmentation\img_LRsumary4357\img_LRsumary\0058_01153_b.jpg"

dataset = LicensePlateDataset(LR_path, HR_path, (256,256))
LR_sample = Image.open(LR_sample_path)
LR_transform = dataset.transforms(LR_sample).unsqueeze(0).to("cuda")

model = UNET(3,3)
model.load_state_dict(torch.load(r"D:\LPD_LPR_SR_Proposal\model\SR\UNET\model_weights.pth"))
model.to("cuda")

output = Output(LR_transform, model)
HR_output = output.inference()
plt.subplot(1,2,1)
plt.imshow(HR_output)
plt.subplot(1,2,2)
plt.imshow(LR_transform.squeeze(0).permute(1,2,0).cpu().numpy())
plt.show()

