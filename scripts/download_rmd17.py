import os
from urllib.request import urlretrieve
import zipfile
import tarfile

# Define paths
DOWNLOAD_PATH = "./data/rmd17_archive.zip"
EXTRACT_DIR = "./data/rmd17"
NESTED_ARCHIVE_PATH = os.path.join(EXTRACT_DIR, 'rmd17.tar.bz2')
FINAL_EXTRACT_DIR = EXTRACT_DIR # Extract into the same rmd17 directory

def download_data():
    """Downloads the rmd17 dataset archive."""
    print("Downloading rmd17 dataset... ", end="")

    # Ensure the data directory exists and handle potential file conflict
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)
    if os.path.exists(EXTRACT_DIR) and not os.path.isdir(EXTRACT_DIR):
        os.remove(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # download rmd17 dataset
    urlretrieve("https://figshare.com/ndownloader/articles/12672038/versions/3", DOWNLOAD_PATH)
    print("Done")

def uncompress_data():
    """Uncompresses the rmd17 dataset archives."""
    # uncompress the main zip file
    print("Uncompressing rmd17 dataset... ", end="")
    with zipfile.ZipFile(DOWNLOAD_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Done")

    # remove the main archive file
    print("Removing archive file... ", end="")
    os.remove(DOWNLOAD_PATH)
    print("Done")

    # uncompress nested tar.bz2
    print(f"Uncompressing nested archive: {NESTED_ARCHIVE_PATH}... ", end="")
    with tarfile.open(NESTED_ARCHIVE_PATH, "r:bz2") as tar_ref:
        tar_ref.extractall(FINAL_EXTRACT_DIR)
    print("Done")

    # remove the nested archive file
    print(f"Removing nested archive file: {NESTED_ARCHIVE_PATH}... ", end="")
    os.remove(NESTED_ARCHIVE_PATH)
    print("Done")

if __name__ == "__main__":
    download_data()
    uncompress_data()
