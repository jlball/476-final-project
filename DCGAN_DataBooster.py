import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

ngf = 64
ndf = 64

class GeneratorNet (nn.Module):
    def __init__(self, input_size):
        super(GeneratorNet, self).__init__()
        self.main = nn.Sequential(
            #THIS SEQUENCE TAKEN FROM PYTORCH TUTORIAL
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, x):
        return self.main(x)

class DiscriminatorNet (nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.main = nn.Sequential(
            #THIS SEQUENCE TAKEN FROM PYTORCH TUTORIAL
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def BoostImageDataset(data_ldr, epochs, num_of_imgs):
    #Check if a GPU is available for use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Initialize the generator and discriminator models, send to appropriate device
    #Specify size of noise vector used as input for generator model
    gen_input_size = 100
    Generator = GeneratorNet(gen_input_size).to(device)
    Discriminator = DiscriminatorNet().to(device)

    #Initialize the weights of each model normally
    Discriminator.apply(weights_init)
    Generator.apply(weights_init)

    #setup binary cross entropy loss function
    loss = nn.BCELoss()

    #naming convention for labels for training
    real = 1
    fake = 0

    #Adam optimizer paramters, as suggested by original DCGAN paper:
    lr = .0002
    beta1 = 0.5

    #Initialize optimizers for both networks
    DiscriminatorOpt = optim.Adam(Discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    GeneratorOpt = optim.Adam(Generator.parameters(), lr=lr, betas=(beta1, 0.999))


    G_losses = []
    D_losses = []
    print("======BEGIN TRAINING DataBoostDCGAN======")
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        for index, batch in enumerate(data_ldr, 0):

            #Train Discriminator network
            #Real Data:
            #zero gradients
            Discriminator.zero_grad()

            #create image tensor and corresponding tensor of real labels
            real_data = batch[0].to(device)
            r_size = real_data.size(0)
            real_labels = torch.full((r_size,), real, dtype=torch.float, device=device)

            #Train Discrimiantor on real data, backpropagate
            y = Discriminator(real_data).view(-1)
            Discriminator_loss_real = loss(y, real_labels)
            Discriminator_loss_real.backward()

            #Fake Data:
            noise = torch.randn(r_size, gen_input_size, 1, 1, device=device)

            #Generate fake image data w/ generator:
            fake_data = Generator(noise)
            #Tensor of labels indicating fake_data is fake:
            fake_labels = torch.full((r_size,), fake, dtype=torch.float, device=device)

            #Train on fake data:
            y = Discriminator(fake_data.detach()).view(-1)
            Discriminator_loss_fake = loss(y, fake_labels)
            Discriminator_loss_fake.backward()

            Discriminator_loss_total = Discriminator_loss_fake + Discriminator_loss_real
            DiscriminatorOpt.step()

            #Train Generator Network:
            Generator.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            y = Discriminator(fake_data).view(-1)
            # Calculate G's loss based on this output
            Generator_loss = loss(y, real_labels)
            # Calculate gradients for G
            Generator_loss.backward()
            GeneratorOpt.step()

            G_losses.append(Generator_loss.item())
            D_losses.append(Discriminator_loss_total.item())

            if index % 50 == 0:
                print("Loss:", Discriminator_loss_fake.item(), Discriminator_loss_real.item())

    torch.no_grad()

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()  

    return Generator(torch.randn(num_of_imgs, gen_input_size, 1, 1, device=device)).detach().cpu()







