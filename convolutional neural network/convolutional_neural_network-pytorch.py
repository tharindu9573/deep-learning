import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hyper parameters
learning_rate = 0.01
batch_size = 4
num_epoches = 5

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    './root', download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    './root', download=True, train=False, transform=transform)

# data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

samples, labels = next(iter(train_loader))

# classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imgshow(img: torch.tensor):  # show iamge
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter)
print(labels[0])
imgshow(torchvision.utils.make_grid(images))


class CNN(nn.Module):  # setup network
    def __init__(self):
        super().__init__()
        # setup layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# model
model = CNN().to(device)

# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# forward
mode = 'test'
if mode == 'train':
    num_iterations = len(train_loader)
    for epoch in range(num_epoches):
        for i, (samples, labels) in enumerate(train_loader):

            images = samples.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2000 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epoches}, Step {i}/{num_iterations}, Loss {loss.item():.4f}")

    print('Finished training !')
    PATH = 'cnn.pth'
    torch.save(model.state_dict(), PATH)

# evaluate
with torch.no_grad():
    n_samples = 0
    n_correct = 0

    loaded_model = CNN().to(device)
    FILE = 'cnn.pth'
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = loaded_model(images)
        _, predicted = torch.max(outputs, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc}%")
