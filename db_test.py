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
    def __init__(self, imgs, num_imgs):
        self.img_labels = cat((full((num_imgs,), 0), full((num_imgs,), 1)), 0)
        self.imgs = imgs
        print("IMGS SIZE:", self.imgs.size())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        sample = (image, label)
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

num_of_images = 5000

raw_imgs = BoostImageDataset(raw_ldr, 150, num_of_images)
processed_imgs = BoostImageDataset(processed_ldr, 150, num_of_images)

total_dataset = DCGANImageDataset(cat((raw_imgs, processed_imgs), 0), num_of_images)
total_ldr = DataLoader(total_dataset, batch_size = 100, shuffle = True)

raw_imgs = vutils.make_grid(raw_imgs, padding=2, normalize=True)
processed_imgs = vutils.make_grid(processed_imgs, padding=2, normalize=True)

# plt.imshow(np.transpose(raw_imgs, (1, 2, 0)))
# plt.show()
# plt.imshow(np.transpose(processed_imgs, (1, 2, 0)))
# plt.show()


trans = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = ImageFolder("/home/jlball/Desktop/Machine Learning/Final Project/images", transform=trans)
train_ldr = total_ldr

acc_train_ldr = DataLoader(train_data)

input_dim = 12544
layer1 = 4000
layer2 = 1000
layer3 = 200
out_dim = 2
filter1 = 16
filter2 = 16

def compute_accuracy(onTrainingData=False):
    net.eval()
    no_grad()
    num_correct = 0
    counter = 0
    if onTrainingData:
        ldr = acc_train_ldr
    for (index, data) in enumerate(ldr):
                pred = net(data[0].cuda())
                pred = pred.argmax()
                counter += 1
                if pred.item() == data[1]:
                    num_correct += 1

    accuracy = num_correct / (ldr.__len__())
    print("Accuracy: ", accuracy)
    set_grad_enabled(True)
    net.train()


net = Sequential(
    Conv2d(3, filter1, kernel_size=2, stride=1, padding=0),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d(filter1, filter2, kernel_size=4, stride=1, padding=0),
    ReLU(),
    #MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(input_dim, layer1),
    ReLU(),
    Linear(layer1, layer2),
    ReLU(),
    Linear(layer2, layer3),
    ReLU(),
    Linear(layer3, out_dim),
)

x = randn(1, 3, image_size, image_size)
for layer in net:
     x = layer(x)
     print(x.size())

#push model to GPU
net.cuda()

#Number of epochs
epochs = 25

#Loss function
loss = CrossEntropyLoss()
learn_rate = 0.001

#Setting up the Adam optimizer
adam = Adam(net.parameters(), lr = learn_rate)

#Train the model
for epoch in range(0, epochs):
    print ("START EPOCH: ", epoch + 1)
    for (batch_idx, batch) in enumerate(train_ldr):
        y = net(batch[0].cuda())
        loss_value = loss(y, batch[1].cuda())

        net.zero_grad()
        adam.zero_grad()
        loss_value.backward()

        adam.step()

        print("loss:", loss_value.item())
    compute_accuracy(onTrainingData=True)
compute_accuracy(onTrainingData=True)
