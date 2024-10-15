import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import PetDataset
from models import VAE

os.mkdir('pretrained_vae', exist_ok=True)

dataset = PetDataset('images', type='train', only_use_british_shorthair=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

def loss_function(recon_x, x, mu, logvar, kld_weight=0.000025):
    recon_loss = F.mse_loss(recon_x, x)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss

def get_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

def get_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(latent_dim=1024).to(device)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)

num_epochs = 1000
writer = SummaryWriter("runs/vae")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0
    progress = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs}')

    for batch_idx, data in enumerate(progress):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, kld_weight=0.00025)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress.set_postfix({'Loss': loss.item(), 'Recon Loss': recon_loss.item(), 'KLD Loss': kld_loss.item()})

    avg_loss = train_loss / len(dataloader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

    scheduler.step(avg_loss)

    writer.add_scalar('Loss/total', avg_loss, epoch)
    writer.add_scalar('Loss/recon', recon_loss.item(), epoch)
    writer.add_scalar('Loss/KLD', kld_loss.item(), epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    model.eval()
    with torch.no_grad():
        sample = data
        recon, _, _ = model(sample)
        comparison = torch.cat([sample.cpu(), recon.cpu()])
        img_grid = make_grid(comparison, nrow=8, normalize=True, scale_each=True)
        writer.add_image('Reconstruction', img_grid, epoch)

    if epoch % 100 == 0:
        if not os.path.exists('pretrained_vae'):
            os.makedirs('pretrained_vae')
        torch.save(model.state_dict(), f'pretrained_vae/epoch_{epoch}.pth')

torch.save(model.state_dict(), 'pretrained_vae/epoch_final.pth')
writer.close()

print("Training completed!")