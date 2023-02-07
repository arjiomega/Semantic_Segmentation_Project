import pandas as pd
from pathlib import Path
import warnings

from config import config
from package import utils


# for processing tar.gz files
import requests
import tarfile

warnings.filterwarnings("ignore")

def elt_data():
    """Extract, Load, and Transform Data Assets
    NOTE:
    1. Download tar.gz files from config urls
    2. load pandas dataframe from annotations/list.txt
    """

    IMG_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    LABEL_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

    # Extract and Load
    img_load = requests.get(config.IMG_URL)
    label_load = requests.get(config.LABEL_URL)

    print("Downloading images.tar.gz ...")
    with open(Path(config.DATA_DIR,"images.tar.gz"),"wb") as file_path:
        file_path.write(img_load.content)

    print("Downloading annotations.tar.gz ...")
    with open(Path(config.DATA_DIR,"annotations.tar.gz"),"wb") as file_path:
        file_path.write(label_load.content)

    print("Extracting images.tar.gz ...")
    with tarfile.open(Path(config.DATA_DIR,"images.tar.gz"),"r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)

    print("Extracting annotations.tar.gz ...")
    with tarfile.open(Path(config.DATA_DIR,"annotations.tar.gz"),"r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)

    # Transform
    # <INSERT HERE>


    # Extract + Load
    # projects = pd.read_csv(config.PROJECTS_URL)
    # tags = pd.read_csv(config.TAGS_URL)
    # projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    # tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # # Transform
    # df = pd.merge(projects, tags, on="id")
    # df = df[df.tag.notnull()]  # drop rows w/ no tag
    # df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)



    #logger.info("✅ Saved data!")

if __name__ == "__main__":
    elt_data()