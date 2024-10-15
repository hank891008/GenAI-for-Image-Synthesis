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
from models import AE

os.makedirs('results_ae', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PetDataset('images', type='test', only_use_british_shorthair=True)
dataloader = DataLoader(dataset, batch_size=1)

model = AE().to(device)
model.load_state_dict(torch.load('pretrained_ae/epoch_final.pth'))
model.eval()

psnr = PSNR().to(device)
ssim = SSIM().to(device)
lpips = LPIPS().to(device)
with torch.no_grad():
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    for i, img in enumerate(dataloader):
        img = img.to(device)
        output = model(img)
        output = (output + 1) / 2
        img = (img + 1) / 2
        grid = make_grid(torch.cat([img, output], dim=0), nrow=8)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(f'results_ae/{i}.jpg')
        psnr_vals.append(psnr(output, img).item())
        ssim_vals.append(ssim(output, img).item())
        lpips_vals.append(lpips(output, img).item())
    print(f'PSNR: {np.mean(psnr_vals):.2f}, SSIM: {np.mean(ssim_vals):.2f}, LPIPS: {np.mean(lpips_vals):.2f}')
        
