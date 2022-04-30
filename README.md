# ML-Object-Identifier
This project is an image classifier, and is capable of classifying between arbitrary types of images.

## Dataset
I pulled our dataset from multiple sources, but in theory, it doesn't matter where you get the images from, or which images you use.

[My dataset can be downloaded here](https://mega.nz/file/ztVkHbZL#kiKhmrf-5lgARhVNqUBdcFmr2qW_PM9aObVRnXgyA6o)

The directory where the images are located should be called "images". Some test images are also provided, to test the capabilities of the model.

## Usage

imageClassif.py is the main script. It has a few arguments:

`-np` - skip image processing (for when you've already processed the images)

`-ep` - specify the number of epochs

`-sv` - specify the save name

`-ld` - load the model with the passed save name

`-lr` - learning rate. Use this to specify the order of magnitude. IE, -3 would be 1e-3

Example: `python imageClassif.py -np -ep100 -svMediumModel -lr-5`

## Components

bbid.py is a web scraping script, intended for use in building a dataset. An example script, scrape.sh, uses this tool.

image_processing.py is my image processing library, which we use to manipulate and format the image for use with the CNN. This also handles most of the file manipulation.

imageClassif.py is the "main" script, as explained in the usage section.

dataviewer.py is used to display information about the loss and accuracy across epochs. You must provide the model's save name that you want to load.
