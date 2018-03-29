"""model.py"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, self.hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, image):
        hidden = self.encode(image)
        out = self.decode(hidden)
        return out, hidden.view(image.size(0), -1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, 4*4*1024, 1)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        out = self.fc(z)
        out = out.view(-1, 1024, 4, 4)
        out = self.conv(out)

        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
