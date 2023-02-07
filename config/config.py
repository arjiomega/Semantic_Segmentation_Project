from pathlib import Path
import logging

#Directories
## __file__ -> /home/rjomega/github_new/E2E_PROJECTS/Semantic_Segmentation/config/config.py
## BASE_DIR -> /home/rjomega/github_new/E2E_PROJECTS/Semantic_Segmentation
## CONFIG_DIR -> /home/rjomega/github_new/E2E_PROJECTS/Semantic_Segmentation/config

BASE_DIR = Path(__file__).parent.parent.absolute()

CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create directories
## parents -> Create parent directory if they do not exist
## exist_ok -> if its okay if the directory already exists, if true then no error raised
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets directory
IMG_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
LABEL_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"


# Logging