import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#data
transform = torchvision.transforms.ToTensor()
trainset = torchvision.datasets.FER2013('./data', 'train', transform=transform)
testset = torchvision.datasets.FER2013('./data', 'test', transform=transform)

batch = 64

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=True)

class FERecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            #1,48,48
            nn.Conv2d(1, 16, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            #16,48,48
            nn.MaxPool2d((2,2)),
            #16,24,24
            nn.Conv2d(16, 32, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            #32,24,24
            nn.MaxPool2d((2,2)),
            #32,12,12
            nn.Conv2d(32, 64, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            #64,12,12
            nn.MaxPool2d((2,2)),
            #64,6,6
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,7)
        )
    
    def forward(self, x):
        return self.layers(x)
    
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

    print("Loss: %.4f Test accuracy %.2f%%" % (total_loss / batches, correct / total *100))
    print()
    return correct / total *100

    
# model = FERecognizer().to(device)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# best = 0
# epochs = 300
# for epoch in range(epochs):
#     print(best)
#     train(epoch)
#     accuracy = test()
#     if accuracy > best:
#         best = accuracy
#         torch.save(model.state_dict(), 'fer2013_2.pth')

model = FERecognizer()
saved = torch.load('fer2013.pth')
model.load_state_dict(saved)
model.eval()

labels_dict = {
    0:'Angry',
    1:'Disgusted',
    2:'Afraid',
    3:'Happy',
    4:'Sad',
    5:'Surprised',
    6:'Neutral'
}

for num in range(30):
    index = random.randint(0, testset.__len__())
    # X = torch.tensor(testset.__getitem__(index), dtype=torch.float32).permute(0,3,1,2)
    h, y = testset.__getitem__(index)
    pred = model(h.unsqueeze(0))
    print('Pred:',labels_dict[pred.argmax().sum().item()])
    print('Real:',labels_dict[y])
    plt.imshow(h.squeeze(), cmap="gray")
    plt.show()