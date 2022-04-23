from os import listdir
from os.path import isfile, join
import os, sys
from PIL import Image # we use PIL to do our image processing, because it is very good.
import PIL

from alive_progress import alive_bar # progress bar is pretty important here

# Suppress warnings, because they make our progress bars look less cool
import warnings
warnings.filterwarnings("ignore")


# CONSTANTS :
# Here is where we define the size of the output images
RESIZE = (50, 50)
# folder where the formatted images will be saved
FORMATED_PATH = "imagesets\\"
# folder where our unaltered image directories are located
IMAGE_FOLDER = "images\\"
# name of the individual image directories
IMAGES_PATHS = ["cats", "dogs", "snails", "msc"]

# returns a list of images (files) in the provided directory
def get_images(dir):
    image_paths = []
    for filename in os.listdir(dir):
        image_paths.append(os.path.join(dir, filename))
    return image_paths

# formats the image to better suit our purposes
def format_image(path, num):
    try:
        im = Image.open(path)
        im = im.resize(RESIZE) # resize to be small
        im = im.convert('L') # convert to grayscale
        f, e = os.path.splitext(path) # remove file extension
        newpath = FORMATED_PATH + f.split('\\')[1] + "\\" + str(num) + ".jpg" # create new path to save at
        try:
            im.save(newpath, 'JPEG')
        except OSError as e:
            print(e)
    except (PIL.UnidentifiedImageError, OSError):
        #print("Unknown image, skipping")
        pass


def format_all_images():
    try:
        os.mkdir(FORMATED_PATH)
    except FileExistsError:
        pass
    image_paths = []
    images_count = 0
    for t in IMAGES_PATHS:
        try:
            os.mkdir(FORMATED_PATH + t)
        except FileExistsError:
            pass
        ims = get_images(IMAGE_FOLDER + t)
        image_paths.append(ims)
        images_count += len(ims)
    p = "images"
    pc = 0
    with alive_bar(images_count, title=f'Processing {p}', length = 50, bar="filling") as bar:
        for t in image_paths:
            p = IMAGES_PATHS[pc]
            current = 0
            for i in t:
                format_image(i, current)
                current += 1
                bar()
            print(p + " images processed")
            pc += 1

if __name__ == "__main__":
    format_all_images()
