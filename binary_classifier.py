from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import Sequential, ReLU, Linear, Flatten, CrossEntropyLoss, Conv2d, MaxPool2d
from torch.optim import Adam
from torch import randn, no_grad, set_grad_enabled
from DCGAN_DataBooster import BoostImageDataset
import matplotlib.pyplot as plt

res = 85

trans = transforms.Compose([
    transforms.Resize((128, res)),
    transforms.ToTensor()
])

train_data = ImageFolder("/home/jlball/Desktop/Machine Learning/Final Project/images", transform=trans)
test_data = ImageFolder("/home/jlball/Desktop/Machine Learning/Final Project/test_images", transform=trans)


train_ldr = DataLoader(train_data, batch_size=120, shuffle=True)
test_ldr = DataLoader(test_data, shuffle=False)
acc_train_ldr = DataLoader(train_data)

input_dim = 37440
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
    else:
        ldr = test_ldr
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
    return accuracy

#Define the binary classifier network
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

x = randn(1, 3, 128, res)
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

loss_plot = []
acc_plot = []
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
        loss_plot.append(loss_value.item())
    acc_plot.append(compute_accuracy())
compute_accuracy(onTrainingData=True)

plt.figure(figsize=(10,5))
plt.title("Binary Classifier Loss over Training")
plt.plot(loss_plot)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.show()  

plt.figure(figsize=(10,5))
plt.title("Binary Classifier Accuracy over Training")
plt.plot(acc_plot)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()  