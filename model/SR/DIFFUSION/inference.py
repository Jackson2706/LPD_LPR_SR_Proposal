import torch
import torchvision
import torch.nn.functional as F
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def calculate_psnr(img1, img2, max_pixel_value=1.0):
#     # Tính Mean Squared Error (MSE)
#     mse = F.mse_loss(img1, img2)
    
#     if mse == 0:  # Nếu hai ảnh giống nhau hoàn toàn
#         return float('inf')
    
#     # Tính PSNR
#     psnr = 10 * torch.log10(max_pixel_value**2 / mse)
#     return psnr.item()

def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    
    mnist = Dataset('train', im_path="D:\DDPM-Pytorch\data64_reshaped")
    mnist_loader = DataLoader(mnist, batch_size=5, shuffle=True, num_workers=2)
   
    j=0
    for root_img in tqdm(mnist_loader):
        original_img=root_img.to(device)
        copy_original=original_img.to(device)
        root_img=root_img.to(device)
        noise_lst=[]
        time_step=[]
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            
            noise_pred = model(root_img, torch.as_tensor(i).unsqueeze(0).to(device))
            # Use scheduler to get x0 and xt-1
            root_img, x0_pred = scheduler.sample_prev_timestep(root_img, noise_pred, torch.as_tensor(i).to(device))
            
            if i==0:
                noise_lst.append(root_img)
                time_step.append(i)
                # denoise_lst.append(x0_pred)
        num=len(noise_lst)
        # root_img = torch.clamp(root_img, -1., 1.).detach().cpu()
        # root_img = (root_img + 1) / 2
        
        psnr_lst=[]
        for i in range(num):
            noise_img=noise_lst[i].to(device)
            # original_img=torch.cat([original_img,noise_img],dim=0)
            # psnr_lst.append(calculate_psnr(copy_original,noise_img))
        

        matrix = torch.clamp(noise_img, -1., 1.).detach().cpu()
        matrix = (matrix + 1) / 2
        grid = make_grid(matrix, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        # if i==0 or i==99:
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples5')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples5'))
        img.save(os.path.join(train_config['task_name'], 'samples5', 'x0_{}.png'.format(j)))
        img.close()
        j+=1
        
        

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
