import torch
import torch.nn as nn


class GradConCAE(nn.Module):
    def __init__(self, in_channel=3):
        super(GradConCAE, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.down(x)
        return self.up(z)


class GradConVAE(nn.Module):
    def __init__(self, in_channel=3):
        super(GradConVAE, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU()
        )

        self.fc11 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)
        self.fc12 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)
        self.fc2 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h4 = self.down(x)
        return self.fc11(h4.view(-1, 3 * 3 * 64)), self.fc12(h4.view(-1, 3 * 3 * 64))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4_d = self.fc2(z)
        recon = self.up(h4_d.view(-1, 64, 3, 3))
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
