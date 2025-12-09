from urllib.request import urlretrieve
import zipfile
import os

print("Downloading rmd17 dataset... ", end="")

# download rmd17 dataset and save to ./data/rmd17
urlretrieve("https://figshare.com/ndownloader/articles/12672038/versions/3", "./data/rmd17")
print("Done")

# unzip the file
print("Unzipping rmd17 dataset... ", end="")
with zipfile.ZipFile("./data/rmd17", "r") as zip_ref:
    zip_ref.extractall("./data/rmd17")
print("Done")

# remove the zip file
print("Removing zip file... ", end="")
os.remove("./data/rmd17")   
print("Done")
