from model import MyLicensePlateModel
from model import UNET

config ={
    "model": {
        "LPD": MyLicensePlateModel(),
        "UNET": UNET()
    },
    "model_weight_path":{
        "UNET": r"D:\LPD_LPR_SR_Proposal\model\SR\UNET\model_weights.pth"
    },
    "LR_folder_path": r"D:\LicensePlate\project_enhance_quality_license_plate\data_augmentation\img_HRsumary4357\img_HRsumary",
    "HR_folder_path": r"D:\LicensePlate\project_enhance_quality_license_plate\data_augmentation\img_LRsumary4357\img_LRsumary"
    
}