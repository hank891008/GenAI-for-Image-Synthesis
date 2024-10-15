import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from dataset import PetDataset
from models import VAE

os.makedirs('results_vae', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 1024

dataset = PetDataset('images', type='test', only_use_british_shorthair=True)
dataloader = DataLoader(dataset, batch_size=1)

psnr = PSNR().to(device)
ssim = SSIM().to(device)
lpips = LPIPS().to(device)

model = VAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load('pretrained_vae/epoch_final.pth'))
model.eval()

with torch.no_grad():
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    for i, img in enumerate(dataloader):
        img = img.to(device)
        output, _, _ = model(img)
        output = (output + 1) / 2
        img = (img + 1) / 2
        grid = make_grid(torch.cat([img, output], dim=0), nrow=8)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(f'results_vae/{i}.jpg')
        psnr_vals.append(psnr(output, img).item())
        ssim_vals.append(ssim(output, img).item())
        lpips_vals.append(lpips(output, img).item())
    print(f'PSNR: {np.mean(psnr_vals):.2f}, SSIM: {np.mean(ssim_vals):.2f}, LPIPS: {np.mean(lpips_vals):.2f}')
        