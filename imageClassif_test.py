import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

import requests
from PIL import Image
from io import BytesIO
import sys
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

from image_processing import format_all_images, convert_array, obtain_dataset_paths

# globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numb_batch = 3
T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])
train_dl = None
val_dl = None

# defining our dataset
class image_dataset(torch.utils.data.Dataset):
    path_dict = {}
    size = 0
    def __init__(self, d):
        torch.utils.data.Dataset.__init__(self)
        path_dict = d
        for t in path_dict:
            for i in t:
                size += 1
    def __getitem__(self, idx):
        sum = 0
        key = 0
        for t in path_dict:
            if idx - sum > len(t):
                sum += len(t)
                key += 1
            else:
                path = t[idx - sum]
                label = path_dict
                return convert_array(path), label

    def __len__(self):
        return size




def define_data(dset):
    train_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=True, transform=T)
    val_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=True, transform=T)

    #train_dl = torch.utils.data.DataLoader(train_data, batch_size = numb_batch)
    train_dl = torch.utils.data.DataLoader(dset, batch_size = numb_batch)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size = numb_batch)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
#dataiter = iter(train_dl)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total

def train(numb_epoch=3, lr=1e-3):
    print("Hello")
    accuracies = []
    cnn = create_lenet()
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    plt.plot(accuracies)
    return best_model

if __name__ == "__main__":
    print("Starting program. . .")
    # run paremeters
    process_images = True
    for arg in sys.argv:
        if arg == "-np": # code for "no (image) processing"
            process_images = False

    if process_images:
        format_all_images()
    else:
        print("Skipping image formating stage")


    formated_paths = obtain_dataset_paths()
    dset = image_dataset(formatted_paths)
    define_data(dset)

    lenet = train(20)
