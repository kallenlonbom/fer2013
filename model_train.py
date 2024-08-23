import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from model import FERecognizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data
transform = torchvision.transforms.ToTensor()
trainset = torchvision.datasets.FER2013('./', 'train', transform=transform)
testset = torchvision.datasets.FER2013('./', 'test', transform=transform)

batch = 64

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=True)

# train sequence
def train(epoch):
    model.train()
    total_loss = 0
    batches = trainloader.__len__()
    for x, labels in trainloader:
        x, labels = x.to(device), labels.to(device)
        pred = model(x)
        loss = loss_fn(pred, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch %d Loss: %.4f%%" % (epoch, total_loss / batches))

# test sequence
def test():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    batches = testloader.__len__()
    for x, labels in testloader:
        x, labels = x.to(device), labels.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, labels).item()
        correct += (torch.argmax(pred, 1) == labels).float().sum().item()
        total += labels.__len__()

    print("Validation loss: %.4f Accuracy: %.2f%%" % (total_loss / batches, correct / total *100))
    print()
    return correct / total *100

    
model = FERecognizer().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training loop, save model state dictionary with highest validation accuracy
best = 0
epochs = 300
for epoch in range(epochs):
    print("Best accuracy: %.2f%%" % best)
    train(epoch)
    accuracy = test()
    if accuracy > best:
        best = accuracy
        torch.save(model.state_dict(), 'fer2013_new.pth')