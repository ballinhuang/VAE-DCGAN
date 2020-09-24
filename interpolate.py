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

parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', default='', help='enables cuda')
parser.add_argument('--netE', required=True,
                    help="path to NetE (to continue training)")
parser.add_argument('--netG', required=True,
                    help="path to NetG (to continue training)")
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--load', action='store_true', default=False,
                    help='')

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
if opt.load == False:
    dataset = dset.ImageFolder(root=opt.dataroot+'new_img_celeba',
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

test_dataset = dset.ImageFolder(root=opt.dataroot+'new_img_celeba_test',
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

device = torch.device(opt.cuda) if opt.cuda != '' else torch.device("cpu")
nc = 3
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
imageSize = int(opt.imageSize)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


NetE = Encoder(imageSize, nc, ngf, nz).to(device)
NetG = Decoder(nc, ngf, nz).to(device)

Sampler = Sampler().to(device)

NetE.apply(weights_init)
NetG.apply(weights_init)

# load weights
NetE.load_state_dict(torch.load(opt.netE, map_location=opt.cuda))
NetG.load_state_dict(torch.load(opt.netG, map_location=opt.cuda))

NetE.eval()
NetG.eval()

# 21 attributes
attributes = [
    '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair',
    'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mustache',
    'No_Beard',  'Receding_Hairline', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Hat', 'Wearing_Lipstick', 'Young'
]

torch.set_grad_enabled(False)

attributes_z = [torch.zeros([100, 1, 1])] * len(attributes)
attributes_c = [0] * len(attributes)

attributes_vector = [torch.zeros([100, 1, 1])] * len(attributes)

if opt.load == False:
    for i, (images, target) in enumerate(dataloader):

        images = images.to(device)

        mu, logvar = NetE(images)
        zs = Sampler([mu, logvar], device)

        for x in range(len(target)):
            attr = target[x].item()
            attributes_c[attr] += 1
            attributes_z[attr] = torch.add(
                attributes_z[attr], zs[x].mul_(0.01).cpu())

        print('[%d/%d]' % (i, len(dataloader)))

    for i in range(len(attributes)):
        tmp = torch.zeros([100, 1, 1])
        tmpc = 0
        for j in range(len(attributes)):
            if j != i:
                tmp = torch.add(tmp, attributes_z[j])
                tmpc += attributes_c[j]
        tmp = torch.mul(torch.div(tmp, tmpc), 100)

        attributes_vector[i] = torch.sub(
            torch.mul(torch.div(attributes_z[i], attributes_c[i]), 100), tmp)

    for i in range(len(attributes)):
        torch.save(attributes_vector[i], '%s/%s.pth' %
                   (opt.outf, attributes[i]))
else:
    for i in range(len(attributes)):
        attributes_vector[i] = torch.load('%s/%s.pth' %
                                          (opt.outf, attributes[i]))

for i in range(len(attributes)):
    attributes_vector[i] = attributes_vector[i].to(device)

for i, (images, target) in enumerate(test_dataloader):
    images = images.to(device)

    vutils.save_image(images, '%s/real.png' %
                      (opt.outf), normalize=True)

    mu, logvar = NetE(images)
    z = Sampler([mu, logvar], device)


    rec_real = NetG(z)
    vutils.save_image(rec_real, '%s/rec_real.png' %
                      (opt.outf), normalize=True)

    for i in range(len(attributes)):
        rec_real = NetG(torch.add(z, torch.mul(
            attributes_vector[i], opt.gamma)))
        vutils.save_image(rec_real, '%s/rec_real_%s.png' %
                          (opt.outf, attributes[i]), normalize=True)
