import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, bottlencek_dim=32):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # conv 0
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), # conv 1
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), # conv 2
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # conv 3
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), # conv 4
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # conv 5
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # conv 6
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # conv 7
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # conv 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, bottlencek_dim, kernel_size=8, stride=1, padding=0), # conv 9
        )

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):
    def __init__(self, bottlencek_dim=32):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(bottlencek_dim, 32, kernel_size=8, stride=1, padding=0), # conv 9d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), # conv 8d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), # conv 7d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # conv 6d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), # conv 5d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # conv 4d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # conv 3d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), # conv 2d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), # conv 1d
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1), # conv 0d
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, bottlencek_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(bottlencek_dim=bottlencek_dim)
        self.decoder = Decoder(bottlencek_dim=bottlencek_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x