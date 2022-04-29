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
import random
from image_processing import format_all_images, convert_array, obtain_dataset_paths
import image_processing
from alive_progress import alive_bar # progress bar is pretty important here

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
    path_l = []
    def __init__(self, d):
        torch.utils.data.Dataset.__init__(self)
        self.path_dict = d
        for t in self.path_dict:
            for i in self.path_dict[t]:
                self.size += 1
                self.path_l.append([i,t])

        random.shuffle(self.path_l)

    def __getitem__(self, idx):
        return convert_array(self.path_l[idx][0]), image_processing.IMAGES_PATHS.index(self.path_l[idx][1])
        sum = 0
        key = 0
        itr = 0
        for t in self.path_dict:
            if idx - sum > len(self.path_dict[t]):
                sum += len(self.path_dict[t])
                key += 1
                itr +=1
            else:
                path = self.path_dict[t][idx - sum]
                label = itr
                return convert_array(path), label.to(device)

    def __len__(self):
        return self.size




def define_data(dset, valset):
    #train_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=True, transform=T)
    #val_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=True, transform=T)

    global train_dl
    global val_dl
    #train_dl = torch.utils.data.DataLoader(train_data, batch_size = numb_batch)
    train_dl = torch.utils.data.DataLoader(dset, batch_size = numb_batch)
    val_dl = torch.utils.data.DataLoader(valset, batch_size = numb_batch)
    #val_dl = torch.utils.data.DataLoader(val_data, batch_size = numb_batch)

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
        nn.Conv2d(3, 6, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(6, 3, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(5),
        nn.Flatten(),
        nn.Linear(81, 50),
        nn.ReLU(),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, 4)
    )
    return model

def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        if len(images) < 3:
            continue
        x = model(images.float().to(device))
        value, pred = torch.max(x,1)
        pred = pred.data
        total += x.size(0)
        correct += torch.sum(pred.to(device) == labels.to(device))
        #print(str(correct))
    return correct*100./total

def train(numb_epoch=3, lr=1e-5):
    global train_dl
    global val_dl
    print("Hello")
    accuracies = []
    cnn = create_lenet()
    cnn.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    for epoch in range(numb_epoch):
        losses = []
        with alive_bar(len(train_dl), title=f'Training epoch {epoch}', length = 40, bar="filling") as bar:
            for i, (images, labels) in enumerate(train_dl):
                if len(images) < 3:
                    #print("Wrong length image, continuing")
                    bar()
                    continue
                images.to(device)
                labels.to(device)

                #print("Image labels : " + str(labels))
                optimizer.zero_grad()
                #print("Shape of images" + str(np.shape(images)))
                pred = cnn(images.float().to(device))
                labels = torch.tensor(np.asarray(labels))
                #print("Lengths of things : " + str(len(pred)) + " lbl " + str(len(labels)))
                loss = cec(pred.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                losses.append(loss)
                bar()
        accuracy = float(validate(cnn.to(device), val_dl))
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


    formated_paths, valpaths = obtain_dataset_paths()
    dset = image_dataset(formated_paths)
    valset = image_dataset(valpaths)
    define_data(dset, valset)

    lenet = train(20)
