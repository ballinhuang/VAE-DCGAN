import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy


class Encoder(nn.Module):
    def __init__(self, imageSize, nc, ngf, nz):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 1, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l_mu = nn.Conv2d(ngf * 8, nz, 4)
        self.l_var = nn.Conv2d(ngf * 8, nz, 4)

    def forward(self, input):
        output = self.encoder(input)
        mu = self.l_mu(output)
        logvar = self.l_var(output)
        return mu, logvar


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, input, device):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mu)


class Decoder(nn.Module):
    def __init__(self,  nc, ngf, nz):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.decoder(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, imageSize, nc, ndf, nz):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l_layer = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        )
        self.sigmoid_output = nn.Sequential(
            nn.BatchNorm2d(ndf * 8, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, mode=''):
        if mode == 'tilde':
            features = self.main(input)
            output = self.l_layer(features)
            return output.view(len(output), -1)
        else:
            features = self.main(input)
            output = self.l_layer(features)
            output = self.sigmoid_output(output)
            return output.view(-1, 1).squeeze(1)
