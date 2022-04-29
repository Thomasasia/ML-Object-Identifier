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
import os
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import random
from image_processing import format_all_images, convert_array, obtain_dataset_paths
import image_processing
from alive_progress import alive_bar # progress bar is pretty important here
import pickle
import matplotlib.image as mpimg

DATA_PATH = "data\\"

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
    # LIST is set to true if the dataset is build off of a list, instead of a dictionary
    def __init__(self, d, LIST = False):
        torch.utils.data.Dataset.__init__(self)
        if LIST:
            path_l = d
        else:
            # build the image list based on the provided dictionary data bins
            self.path_dict = d
            for t in self.path_dict:
                for i in self.path_dict[t]:
                    self.size += 1
                    self.path_l.append([i,t]) # image path in [0], label in [1]

            # shuffle the images, so that the same ones are not made subsequent
            random.shuffle(self.path_l)

    def __getitem__(self, idx):
        return convert_array(self.path_l[idx][0]), image_processing.IMAGES_PATHS.index(self.path_l[idx][1])

    def __len__(self):
        return len(self.path_l)




def define_data(dset, valset):

    # load the data for use in the CNN
    global train_dl
    global val_dl
    train_dl = torch.utils.data.DataLoader(dset, batch_size = numb_batch)
    val_dl = torch.utils.data.DataLoader(valset, batch_size = numb_batch)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# here we define out cnn
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

# function to validate the model
def validate(model, data):
    total = 0
    correct = 0
    cec = nn.CrossEntropyLoss()
    losses = []
    for i, (images, labels) in enumerate(data):
        if len(images) < 3:
            continue
        x = model(images.float().to(device))
        value, pred = torch.max(x,1)
        pred = pred.data
        total += x.size(0)
        correct += torch.sum(pred.to(device) == labels.to(device))
        loss = cec(x.to(device), labels.to(device))
        losses.append(loss)
    return correct*100./total, losses

def train(numb_epoch=3, lr=1e-4, save_name = ""):
    global train_dl
    global val_dl
    print("Hello")
    accuracies = []
    cnn = create_lenet()
    cnn.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    total_losses = []
    train_acc = []
    val_losses = []
    for epoch in range(numb_epoch):
        losses = []
        correct = 0
        total = 0
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

                v, p = torch.max(pred,1)
                p = p.data
                total += pred.size(0)
                correct += torch.sum(p.to(device) == labels.to(device))
                #print("Lengths of things : " + str(len(pred)) + " lbl " + str(len(labels)))
                loss = cec(pred.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                losses.append(loss)
                bar()
        train_acc.append(correct*100./total)
        accuracy, val_loss = validate(cnn.to(device), val_dl)
        accuracy = float(accuracy)
        val_loss = val_loss
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch, "Accuracy :", accuracy, '%')
        val_losses.append(val_loss)
        total_losses.append(losses)
    save_data("train_loss", detensor(total_losses, type="loss"), 1, save_name=save_name)
    save_data("validation_accuracy", detensor(accuracies), 2, save_name=save_name)
    save_data("train_accuracy", detensor(train_acc), 2, save_name=save_name)
    save_data("validation_loss", detensor(val_losses, type="loss"), 1, save_name=save_name)
    save_data("model", best_model, 0, save_name=save_name)
    return best_model


def detensor(data, type = "acc"):
    new_data = []
    for i in data:
        if type == "acc":
            if isinstance(i, float):
                new_data.append(i)
            else:
                new_data.append(i.cpu().data.numpy().argmax())
        elif type == "loss":
            subdata = []
            for g in i:
                subdata.append(g.cpu().data.numpy().argmax())
            new_data.append(subdata)
    return new_data


# save the data externally, so that it can be used in graphs and such
def save_data(name, data, type=0, save_name = ""):
    if type != 0:
        try:
            data = data.tolist()
        except AttributeError:
            pass

    # make a dir in the data directory corrosponding to the name
    dir_path = DATA_PATH + name + "\\"
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    f = image_processing.get_images(dir_path) # actually gets files
    filename = ""
    if save_name == "":
        filename = dir_path + str(len(f))
    else:
        filename = dir_path + save_name
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_model(name = ""):
    files = image_processing.get_images("data\\model\\")
    model = None
    if name == "":
        model = files[-1] # most recent one
    else:
        model = "data\\model\\" + name
    with open(model, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    print("Starting program. . .")
    # run paremeters
    process_images = True
    loading_model = False
    ld_name = ""
    save_name = ""
    epochs = 100
    lr=1e-4
    for arg in sys.argv:
        if arg == "-np": # code for "no (image) processing"
            process_images = False
        if arg[0:3] == "-ld" : # code for load (model)
            loading_model = True.cpu().data.numpy().argmax(i
            ld_name = arg[3:len(arg)]
        if arg[0:3] == "-sv" : # code for save (name)
            save_name = arg[3:len(arg)]
        if arg[0:3] == "-ep" : # code for epoch (count)
            epochs = int(arg[3:len(arg)])
        if arg[0:3] == "-lr" : # code for learning rate
            lr=10 ** int(arg[3:len(arg)])
    if process_images:
        format_all_images()
    else:
        print("Skipping image formating stage")


    formated_paths, valpaths = obtain_dataset_paths()
    dset = image_dataset(formated_paths)
    valset = image_dataset(valpaths)
    define_data(dset, valset)

    if not loading_model:
        print("Training model")
        print("Learning rate : " + str(lr))
        lenet = train(epochs, save_name = save_name, lr = lr)
    else:
        print("Loading model")
        lenet = load_model(ld_name)

    if not process_images:
        format_all_images(only_test = True)
    ims = image_processing.obtain_test_data()
    testset = image_dataset(ims, LIST=True)
    test_dl = torch.utils.data.DataLoader(testset, batch_size = numb_batch)

    ims_tensor = []
    count = 0
    bin = -1
    for i in ims:
        if count == 0:
            bin += 1
            ims_tensor.append([])
        ia = image_processing.convert_array(i)
        ims_tensor[bin].append(ia)
        if count == 2 : count = 0
        else : count += 1
    results = []
    for batch in ims_tensor:
        if len(batch) == 3:
            images = torch.tensor(batch)
            images.to(device)
            pred = lenet(images.float().to(device))
            v, p = torch.max(pred,1)
            p = p.data
            results = results + p.tolist()

    og_test_images = image_processing.get_images(image_processing.TEST_IMAGE_PATH)
    ims_arr = []
    for i in og_test_images:
        ims_arr.append(image_processing.convert_array(i))
    for i in range(len(ims_arr)):
        fig = plt.figure()
        imgplot = plt.imshow(mpimg.imread(og_test_images[i]))
        fig.suptitle(image_processing.IMAGES_PATHS[results[i]], fontsize=30)
        plt.xlabel(image_processing.IMAGES_PATHS[results[i]])
        plt.show()
