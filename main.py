from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from network import Decoder, Discriminator, Encoder, Sampler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    help='cifar10 | mnist | celeba')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=55,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default='', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netE', default='',
                    help="path to NetE (to continue training)")
parser.add_argument('--netG', default='',
                    help="path to NetG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to NetD (to continue training)")
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--gamma", default=4e-5, help='gamma',
                    action="store", type=float)
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'celeba':
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot+'celeba',
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc = 3

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc = 3

elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(opt.imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


device = torch.device(opt.cuda) if opt.cuda != '' else torch.device("cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
imageSize = int(opt.imageSize)
lr = opt.lr
gamma = opt.gamma


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


NetG = Decoder(nc, ngf, nz).to(device)
NetD = Discriminator(imageSize, nc, ndf, nz).to(device)
NetE = Encoder(imageSize, nc, ngf, nz).to(device)
Sampler = Sampler().to(device)

NetE.apply(weights_init)
NetG.apply(weights_init)
NetD.apply(weights_init)

# load weights
if opt.netE != '':
    NetE.load_state_dict(torch.load(opt.netE))
if opt.netG != '':
    NetG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    NetD.load_state_dict(torch.load(opt.netD))

optimizer_encorder = optim.RMSprop(params=NetE.parameters(
), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
optimizer_decoder = optim.RMSprop(params=NetG.parameters(
), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
optimizer_discriminator = optim.RMSprop(params=NetD.parameters(
), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)

data, _ = next(iter(dataloader))
fixed_batch = Variable(data).to(device)
vutils.save_image(fixed_batch,
                  '%s/real_samples.png' % opt.outf,
                  normalize=True)

margin = 0.6
equilibrium = 0.68

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # input
        real_cpu = data[0]
        batch_size = real_cpu.size(0)
        input = Variable(real_cpu).to(device)

        # train with real
        real_class = NetD(input)
        errDis_real = -torch.log(real_class + 1e-5)
        D_x = real_class.mean().item()

        # train with fake
        noise = Variable(torch.randn(batch_size, nz, 1, 1)).to(device)
        noise.normal_(0, 1)
        fake = NetG(noise)
        fake_class = NetD(fake)
        errDis_fake = -torch.log(1-real_class + 1e-5)
        D_G_z1 = fake_class.mean().item()

        # reconstruct
        mu, logvar = NetE(input)
        sample = Sampler([mu, logvar], device)
        rec_real = NetG(sample)
        rec_class = NetD(rec_real)
        errDis_rec = -torch.log(1-rec_class + 1e-5)
        D_G_z2 = rec_class.mean().item()

        l_x = NetD(input, "tilde")
        l_x_tilde = NetD(rec_real, "tilde")
        errDec_MSE = torch.sum(0.5*(l_x - l_x_tilde) ** 2, 1)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss_discriminator = torch.sum(
            errDis_fake) + torch.sum(errDis_real) + torch.sum(errDis_rec)
        loss_decoder = torch.sum(errDec_MSE * gamma) - \
            torch.sum(loss_discriminator*(1-gamma))
        loss_encoder = torch.sum(errDec_MSE) + torch.sum(KLD)

        train_dec = True
        train_dis = True
        up = torch.mean(errDis_real)
        low = torch.mean(0.5*errDis_real+0.5*errDis_rec)
        if up < equilibrium-margin or low < equilibrium-margin:
            train_dis = False
        if up > equilibrium+margin or low > equilibrium+margin:
            train_dec = False
        if train_dec is False and train_dis is False:
            train_dis = True
            train_dec = True

        NetE.zero_grad()
        loss_encoder.backward(retain_graph=True)
        optimizer_encorder.step()

        if train_dec:
            NetG.zero_grad()
            loss_decoder.backward(retain_graph=True)
            optimizer_decoder.step()

        if train_dis:
            NetD.zero_grad()
            loss_discriminator.backward()
            optimizer_discriminator.step()

        print('[%d/%d][%d/%d] loss_discriminator: %.4f loss_decoder: %.4f loss_encoder: %.4f D_x: %.4f D_G_z1: %.4f  D_G_z2: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                  loss_discriminator.item(), loss_decoder.item(), loss_encoder.item(), D_x, D_G_z1, D_G_z2))

    mu, logvar = NetE(fixed_batch)
    sample = Sampler([mu, logvar], device)
    rec_real = NetG(sample)
    vutils.save_image(rec_real, '%s/rec_real_epoch_%03d.png' %
                      (opt.outf, epoch), normalize=True)
    if epoch % 10 == 0:
        torch.save(NetE.state_dict(), '%s/NetE_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(NetG.state_dict(), '%s/NetG_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(NetD.state_dict(), '%s/NetD_epoch_%d.pth' %
                   (opt.outf, epoch))

noise = Variable(torch.randn(batch_size, nz, 1, 1)).to(device)
noise.normal_(0, 1)
rec_noise = NetG(noise)
vutils.save_image(rec_noise, '%s/rec_noise.png' % (opt.outf), normalize=True)
