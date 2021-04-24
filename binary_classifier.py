from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import Sequential, ReLU, Linear, Flatten, CrossEntropyLoss, Conv2d
from torch.optim import Adam
from torch import randn, no_grad, set_grad_enabled

res = 64

trans = transforms.Compose([
    transforms.Resize((res, res)),
    transforms.ToTensor()
])

data = ImageFolder("/home/jlball/Desktop/Machine Learning/Final Project/images", transform=trans)

print(type(data))
train_ldr = DataLoader(data, batch_size=60, shuffle=True)
test_ldr = DataLoader(data, shuffle=False)

input_dim = res * res * 3
layer1 = 4000
layer2 = 1000
layer3 = 200
out_dim = 2

def compute_accuracy():
    net.eval()
    no_grad()
    num_correct = 0
    counter = 0
    for (index, data) in enumerate(test_ldr):
            pred = net(data[0].cuda())
            pred = pred.argmax()
            counter += 1
            if pred.item() == data[1]:
                num_correct += 1

    accuracy = num_correct / (test_ldr.__len__())
    print("Accuracy: ", accuracy)
    set_grad_enabled(True)
    net.train()


net = Sequential(
    Flatten(),
    Linear(input_dim, layer1),
    ReLU(),
    Linear(layer1, layer2),
    ReLU(),
    Linear(layer2, layer3),
    ReLU(),
    Linear(layer3, out_dim),
)

x = randn(1, 3, res, res)
for layer in net:
    x = layer(x)
    print(x.size())

#push model to GPU
net.cuda()


#Number of epochs
epochs = 10

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

compute_accuracy()