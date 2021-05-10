from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.utils as vutils
from torch.nn import Sequential, ReLU, Linear, Flatten, CrossEntropyLoss, Conv2d, MaxPool2d
from torch.optim import Adam
from torch import randn, no_grad, set_grad_enabled, cat, full
from DCGAN_DataBooster import BoostImageDataset
import matplotlib.pyplot as plt
import numpy as np

class DCGANImageDataset (Dataset):
    def __init__(self, raw_imgs, processed_imgs, num_imgs):
        self.img_labels = cat((full((num_imgs,), 0), full((num_imgs,), 1)), 0)
        self.imgs = cat((raw_imgs, processed_imgs), 0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs.iloc[idx]
        label = self.img_labels.iloc[idx]
        sample = {"image": image, "label": label}
        return sample


dataroot_raw = "/home/jlball/Desktop/Machine Learning/Final Project/images_gan/raw"
dataroot_processed = "/home/jlball/Desktop/Machine Learning/Final Project/images_gan/processed"

image_size = 64

dataset_raw = ImageFolder(root=dataroot_raw,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset_processed = ImageFolder(root=dataroot_processed,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

raw_ldr = DataLoader(dataset_raw, batch_size=120, shuffle=True)
processed_ldr = DataLoader(dataset_processed, batch_size=120, shuffle=True)

num_of_images = 64

raw_imgs = BoostImageDataset(raw_ldr, 10, num_of_images)
processed_imgs = BoostImageDataset(processed_ldr, 10, num_of_images)

print(raw_imgs.shape)

raw_imgs = vutils.make_grid(raw_imgs, padding=2, normalize=True)
processed_imgs = vutils.make_grid(processed_imgs, padding=2, normalize=True)

total_dataset = DCGANImageDataset(raw_imgs, processed_imgs, num_of_images)
total_ldr = DataLoader(total_dataset, batch_size = 8, shuffle = True)

plt.imshow(np.transpose(raw_imgs, (1, 2, 0)))
plt.show()
plt.imshow(np.transpose(processed_imgs, (1, 2, 0)))
plt.show()

