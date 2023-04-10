from pathlib import Path
import os
import copy
import math

import pandas as pd
import numpy as np
import cv2

from config import config
import data_utils


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

def initialize() -> tuple[pd.DataFrame,dict]:
    df = pd.read_csv(Path(config.MASK_DIR.parent,'list.txt'), comment='#',header=None, sep=" ")
    df.columns = ['image_name','class_index','specie_index','breed_index']
    df = df.sort_values(by = 'image_name', ascending=True)
    df = df.reset_index(drop=True)

    return df

def load_data(include_test=True,run_pytest=False) -> tuple[data_utils.Load_Dataset,...]:
    df = initialize()

    # Get dict ("image_name": specie_index)
    specie_dict = df.set_index("image_name")["specie_index"].to_dict()
    subclass_dict = df.set_index("image_name")["class_index"].to_dict()

    # LOAD DATA (list of "DogOrCatName.jpg")
    img_inputs, mask_inputs = data2list(config.IMG_DIR,config.MASK_DIR)

    # CLEAN DATA (Remove unlabeled and duplicate data)
    clean_data_ = data_utils.Clean_Data(img_inputs = img_inputs,
                                           mask_inputs = mask_inputs,
                                           specie_dict = specie_dict)
    _,_ = clean_data_.remove_unlabeled_data()

    if run_pytest == False:
        _,_ = clean_data_.remove_duplicate_data()

    img_inputs,mask_inputs = clean_data_.img_ids, clean_data_.mask_ids

    # Split Data
    inp_list = [img.split(".")[0] for img in img_inputs]

    img_list = data_utils.splitter_funcs.class_splitter(inp_list,subclass_dict,'.jpg')
    mask_list = data_utils.splitter_funcs.class_splitter(inp_list,subclass_dict,'.png')

    train_img,train_mask = [],[]
    valid_img,valid_mask = [],[]
    test_img,test_mask   = [],[]

    # iterate over all classes
    for img,mask in zip(img_list,mask_list):

        temp_ds = data_utils.splitter_funcs.dataset_splitter(img_list[img],mask_list[mask],0.9,0.08,0.02,shuffle=True)
        train_img.extend(temp_ds["train"]["img"])
        train_mask.extend(temp_ds["train"]["mask"])
        valid_img.extend(temp_ds["valid"]["img"])
        valid_mask.extend(temp_ds["valid"]["mask"])
        test_img.extend(temp_ds["test"]["img"])
        test_mask.extend(temp_ds["test"]["mask"])

    train_dataset = data_utils.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = train_img,
                                                mask_list= train_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)

    valid_dataset = data_utils.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = valid_img,
                                                mask_list= valid_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)

    test_dataset = data_utils.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = test_img,
                                                mask_list= test_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)


    return train_dataset, valid_dataset, test_dataset


def preprocess_fn(img,mask):

    # temporary
    preprocessed_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    preprocessed_mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)

    preprocessed_img = np.round(preprocessed_img)
    preprocessed_mask = np.round(preprocessed_mask)

    return preprocessed_img, preprocessed_mask