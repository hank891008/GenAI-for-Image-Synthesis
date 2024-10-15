import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024 x 8 x 8
            nn.BatchNorm2d(1024),
            nn.Mish(inplace=True),

        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),  # 1024 x 16 x 16
            nn.BatchNorm2d(1024),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 3 x 256 x 256
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024 x 8 x 8
            nn.BatchNorm2d(1024),
            nn.Mish(inplace=True),
            
            nn.Conv2d(1024, latent_dim * 2, kernel_size=3, stride=1, padding=1) # (2 x latent_dim) x 8 x 8
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 2, 1),  # 1024 x 16 x 16
            nn.BatchNorm2d(1024),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 3 x 256 x 256
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar