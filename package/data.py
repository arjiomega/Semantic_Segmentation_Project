import os
import pandas as pd
import numpy as np
import cv2
import copy
from pathlib import Path
import math
from config import config

from .data_utilities import clean_data,load_dataset


def data2list(img_dir: str,mask_dir: str) -> tuple[list,list]:
    """Get a list of img and mask file names

    Args:
        img_dir (str): complete path directory
        mask_dir (str): complete path directory

    Returns:
        tuple[list,list]: list of images and mask (ex. ["dogname.jpg",...],["dogname.png",...])
    """
    img_inputs = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
    mask_inputs = [mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
    img_inputs = sorted(img_inputs)
    mask_inputs = sorted(mask_inputs)

    return img_inputs, mask_inputs


def split_data(img_inputs: list,mask_inputs: list,specie_dict: dict, include_test=True) -> dict:
    ClassList_dict = {}
    classes = ["background","cat","dog"]

    for class_ in classes:
        specie_index = classes.index(class_)
        ClassList_dict[class_] = {"img" : [img_  for img_  in img_inputs  if specie_dict[(img_.split(".")[0])]  == specie_index],
                                    "mask": [mask_ for mask_ in mask_inputs if specie_dict[(mask_.split(".")[0])] == specie_index]
                                    }

    split_setting = {"train" : 0.7,
                     "valid" : 0.2,
                     "test" : 0.1}
    if not include_test:
        split_setting["valid"] = split_setting["test"]
        del split_setting["test"]

    dataset_list = list(split_setting.keys())
    all_dataset = {}

    prev_length = [0] * len(classes)
    start = [0] * len(classes)
    end = [0] * len(classes)

    for ds in dataset_list:
        temp_img_list = []
        temp_mask_list = []

        for class_ in classes:

            # SKIP: for background because all images have cat or dog (background image list has 0 length)
            if len(ClassList_dict[class_]["img"]) == 0:
                continue

            specie_index = classes.index(class_)

            start = copy.deepcopy(prev_length)
            dataset_length = math.floor( split_setting[ds] * len(ClassList_dict[class_]["img"]))
            end[specie_index] = start[specie_index] + dataset_length

            # slice until the end of the list
            if ds == dataset_list[-1]:
                end[specie_index] = None

            # sample: cat_img_list = ClassList_dict["cat"]["img"] -> ["catName01.jpg", "catName02.jpg", ...]
            start_now = start[specie_index]
            end_now = end[specie_index]
            temp_img_list.extend([img_    for img_   in ClassList_dict[class_]["img"][start_now:end_now]   ])
            temp_mask_list.extend([mask_  for mask_  in ClassList_dict[class_]["mask"][start_now:end_now]  ])

        prev_length = copy.deepcopy(end)
        all_dataset[ds] =   {"img" : temp_img_list,
                             "mask": temp_mask_list}

    return all_dataset

def initialize() -> tuple[pd.DataFrame,dict]:
    df = pd.read_csv(Path(config.MASK_DIR.parent,'list.txt'), comment='#',header=None, sep=" ")
    df.columns = ['image_name','class_index','specie_index','breed_index']
    df = df.sort_values(by = 'image_name', ascending=True)
    df = df.reset_index(drop=True)

    # Get dict ("image_name": specie_index)
    specie_dict = df.set_index("image_name")["specie_index"].to_dict()

    return df, specie_dict

def load_data(include_test=True,run_pytest=False) -> tuple[load_dataset.Load_Dataset,...]:
    df, specie_dict = initialize()

    # LOAD DATA (list of "DogOrCatName.jpg")
    img_inputs, mask_inputs = data2list(config.IMG_DIR,config.MASK_DIR)

    # CLEAN DATA (Remove unlabeled and duplicate data)
    clean_data_ = clean_data.Clean_Data(img_inputs = img_inputs,
                                           mask_inputs = mask_inputs,
                                           specie_dict = specie_dict)
    _,_ = clean_data_.remove_unlabeled_data()

    if run_pytest == False:
        _,_ = clean_data_.remove_duplicate_data()

    img_inputs,mask_inputs = clean_data_.img_ids, clean_data_.mask_ids

    # Split Data
    all_dataset = split_data(img_inputs,mask_inputs,specie_dict, include_test=include_test)

    #Load Dataset
    train_img, train_mask = all_dataset["train"]["img"],  all_dataset["train"]["mask"]
    valid_img, valid_mask = all_dataset["valid"]["img"],  all_dataset["valid"]["mask"]
    test_img, test_mask = all_dataset["test"]["img"],  all_dataset["test"]["mask"]

    train_dataset = load_dataset.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = train_img,
                                                mask_list= train_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)

    valid_dataset = load_dataset.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = valid_img,
                                                mask_list= valid_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)

    test_dataset = load_dataset.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = test_img,
                                                mask_list= test_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)


    return train_dataset, valid_dataset, test_dataset # add train_dataset next time


def preprocess_fn(img,mask):

    # temporary
    preprocessed_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    preprocessed_mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)

    preprocessed_img = np.round(preprocessed_img)
    preprocessed_mask = np.round(preprocessed_mask)

    return preprocessed_img, preprocessed_mask