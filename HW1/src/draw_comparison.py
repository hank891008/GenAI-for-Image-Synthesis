import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PetDataset
from models import AE, VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 1024

dataset = PetDataset('images', type='train', only_use_british_shorthair=True)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=20, pin_memory=True)

ae = AE().eval().to(device)
ae.load_state_dict(torch.load('pretrained_ae/epoch_final.pth', map_location=device))

vae = VAE(latent_dim=latent_dim).eval().to(device)
vae.load_state_dict(torch.load('pretrained_vae/epoch_final.pth', map_location=device))

def show_comparison(data, recon_vae, recon_ae, n=10):
    data = (data + 1) / 2
    recon_vae = (recon_vae + 1) / 2
    recon_ae = (recon_ae + 1) / 2

    data = data.detach().cpu()
    recon_vae = recon_vae.detach().cpu()
    recon_ae = recon_ae.detach().cpu()

    fig, axes = plt.subplots(3, n, figsize=(20, 6))
    for i in range(n):
        axes[0, i].imshow(data[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title('GT')
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_vae[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title('VAE')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(recon_ae[i].permute(1, 2, 0).numpy())
        axes[2, i].set_title('AE')
        axes[2, i].axis('off')

    plt.savefig('comparison.png')

for data in dataloader:
    data = data.to(device)
    recon_ae = ae(data)
    recon_vae, _, _ = vae(data)
    show_comparison(data, recon_vae, recon_ae)
    break
