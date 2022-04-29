import sys
import pickle
from imageClassif import DATA_PATH
import matplotlib.pyplot as plt



def unpickle(path):
    with open(".\\" + path, "rb") as file:
        return pickle.load(file)

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
    print(str(validation_loss))
