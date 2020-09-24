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


NetE = Encoder(imageSize, nc, ngf, nz).to(device)
Sampler = Sampler().to(device)
NetG = Decoder(nc, ngf, nz).to(device)

NetE.apply(weights_init)
NetG.apply(weights_init)

# load weights
if opt.netE != '':
    NetE.load_state_dict(torch.load(opt.netE))
if opt.netG != '':
    NetG.load_state_dict(torch.load(opt.netG))

optimizer_encorder = optim.RMSprop(params=NetE.parameters(
), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
optimizer_decoder = optim.RMSprop(params=NetG.parameters(
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
        # reconstruct
        mu, logvar = NetE(input)
        sample = Sampler([mu, logvar], device)
        rec_real = NetG(sample)

        errDec_MSE = torch.sum(0.5*(input - rec_real) ** 2, 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss_decoder = torch.sum(errDec_MSE)
        loss_encoder = torch.sum(errDec_MSE) + torch.sum(KLD)

        NetE.zero_grad()
        loss_encoder.backward(retain_graph=True)
        optimizer_encorder.step()

        NetG.zero_grad()
        loss_decoder.backward(retain_graph=True)
        optimizer_decoder.step()

        print('[%d/%d][%d/%d] loss_decoder: %.4f loss_encoder: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                  loss_decoder.item(), loss_encoder.item()))

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
