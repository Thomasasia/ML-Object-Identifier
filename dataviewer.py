import sys
import pickle
from imageClassif import DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean



def unpickle(path):
    with open(".\\" + path, "rb") as file:
        return pickle.load(file)

# normalize 0d arrays
def normalize(data):
    new_data = []
    for d in data:
        nd = []
        for i in d:
            nd.append(i.flat[0])
        new_data.append(nd)
    return new_data

def average(data):
    new_data = []
    for d in data:
        new_data.append(mean(d))
    return new_data

def plot_data(data, ylabel, title):
    x = []
    for i in range(len(data)):
        x.append(i)

    fig, ax = plt.subplots()
    ax.plot(x, data)
    fig.suptitle(title, fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.show()

def plot_two(data1, data2, ylabel, title, l1, l2):
    x = []
    for i in range(len(data1)):
        x.append(i)

    fig, ax = plt.subplots()
    p1 = ax.plot(x, data1, c="red", label=l1)
    p2 = ax.plot(x, data2, c="blue", label=l2)
    ax.legend()
    fig.suptitle(title, fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Too few arguments. Add a name for the data to load.")
    name = sys.argv[1]

    print("Loading data from " + DATA_PATH + "*\\" + name)

    modelpath = DATA_PATH + "model\\" + name

    trainlosspath = DATA_PATH + "train_loss\\" + name
    trainaccpath = DATA_PATH + "train_accuracy\\" + name
    validlosspath = DATA_PATH + "validation_loss\\" + name
    validaccpath = DATA_PATH + "validation_accuracy\\" + name

    print("Loading training data...")
    training_accuracy   = unpickle(trainaccpath)
    training_loss       = unpickle(trainlosspath)
    print("Loading validation data...")
    validation_accuracy = unpickle(validaccpath)
    validation_loss     = unpickle(validlosspath)

    plot_data(training_accuracy, "accuracy", "Training accuracy")
    plot_data(validation_accuracy, "accuracy", "Validation accuracy")
    plot_data(training_loss, "loss", "Avg. Training loss")
    plot_data(validation_loss, "loss", "Avg. Validation loss")
    plot_two(training_accuracy, validation_accuracy, "accuracy", "Accuracy", "Training accuracy", "Validation accuracy")
    plot_two(training_loss, validation_loss, "avg. loss", "Avg. Loss", "Avg. Training loss", "Avg. Validation loss")
