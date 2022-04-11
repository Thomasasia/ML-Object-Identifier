# for sleep
import time
# for the errors
from urllib import error
# scraper library
from bing_image_downloader import downloader


# get input
qry = input("enter the search term: ")

count = input("Enter the number of images you need:")

downloader.download(qry, limit=int(count), output_dir="./images/", timeout=20, verbose=True, error_protection = True)
