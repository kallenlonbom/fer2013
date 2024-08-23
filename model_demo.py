import torch
import torchvision
import random
import matplotlib.pyplot as plt
from model import FERecognizer

# data
transform = torchvision.transforms.ToTensor()
testset = torchvision.datasets.FER2013('./data', 'test', transform=transform)

# load pretrained model
model = FERecognizer()
saved = torch.load('fer2013.pth', weights_only = True)
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

# display 10 demo images, predictions, and labels
for num in range(10):
    index = random.randint(0, testset.__len__())
    h, y = testset.__getitem__(index)
    pred = model(h.unsqueeze(0))
    print('Pred:',labels_dict[pred.argmax().sum().item()])
    print('Real:',labels_dict[y])
    print()
    plt.imshow(h.squeeze(), cmap="gray")
    plt.show()