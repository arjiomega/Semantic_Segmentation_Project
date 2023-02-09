import os
import pandas as pd
import numpy as np
import cv2
import copy

import math
from config import config

from .data_utilities import clean_data,load_dataset,dataloader


def data2list(img_dir: str,mask_dir: str):# -> tuple[list,list]:
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

    # add this to test
    assert len(img_inputs) == len(mask_inputs), "img_inputs and mask_inputs do not have the same length"
    assert img_inputs[0].split(".")[0] == mask_inputs[0].split(".")[0]
    print(img_inputs[0].split(".")[0], mask_inputs[0].split(".")[0])

    return img_inputs, mask_inputs


def split_data(img_inputs,mask_inputs,specie_dict, include_test=True):
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


def load_data():
    """preprocess the data (Maybe change name to load??)
        1. Load Data
        1. Clean Data
        3. Split Data
        4. Load Dataset
        LAST. return
    """

    #######################
    # LOAD DATAFRAME
    df = pd.read_csv('annotations/list.txt', comment='#',header=None, sep=" ")
    df.columns = ['image_name','class_index','specie_index','breed_index']
    df = df.sort_values(by = 'image_name', ascending=True)
    df = df.reset_index(drop=True)

    # Get dict ("image_name": specie_index)
    specie_dict = df.set_index("image_name")["specie_index"].to_dict()
    #######################


    # LOAD DATA (list of "DogOrCatName.jpg")
    img_inputs, mask_inputs = data2list(config.IMG_DIR,config.MASK_DIR)

    # CLEAN DATA (Remove unlabeled and duplicate data)
    clean_data = clean_data.Clean_Data(img_inputs = img_inputs,
                                           mask_inputs = mask_inputs,
                                           specie_dict = specie_dict)

    img_inputs,mask_inputs = clean_data.img_ids, clean_data.mask_ids

    # Split Data
    all_dataset = split_data(img_inputs,mask_inputs,specie_dict, include_test=True)


    #Load Dataset
    train_img, train_mask = all_dataset["train"]["img"],  all_dataset["train"]["mask"]
    valid_img, valid_mask = all_dataset["valid"]["img"],  all_dataset["valid"]["mask"]

    train_dataset = load_dataset.Load_Dataset(img_dir = {config.IMG_DIR:train_img},
                                                mask_dir = {config.MASK_DIR,train_mask},
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)

    valid_dataset = load_dataset.Load_Dataset(img_dir = {config.IMG_DIR:valid_img},
                                                mask_dir = {config.MASK_DIR,valid_mask},
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=preprocess_fn,#preprocessing function
                                                fix_mask=True)


    return train_dataset, valid_dataset # add train_dataset next time





def preprocess_fn(img,mask):

    # temporary
    preprocessed_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    preprocessed_mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)

    return preprocessed_img, preprocessed_mask








# """NOTE:
#     1. Consider splitting classes equally
#     2. img_dir and img_name in pandas dataframe, try to make then equal in each element
# """

# img_dir = "images"
# mask_dir = "annotations/trimaps"

# # remove hidden files and not jpg or png
# img_inputs = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
# mask_inputs = [mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
# img_inputs = sorted(img_inputs)
# mask_inputs = sorted(mask_inputs)

# assert len(img_inputs) == len(mask_inputs), "img_inputs and mask_inputs do not have the same length"

# # divide equally to prevent class imbalance in dataset
# ## this one may produce errors again 
# cat_img = img_inputs[:2370]
# dog_img = img_inputs[2370:]

# cat_mask = mask_inputs[:2370]
# dog_mask = mask_inputs[2370:]


# train_img = cat_img[:len(cat_img)//2] + dog_img[:len(dog_img)//2]
# train_mask = cat_mask[:len(cat_mask)//2] + dog_mask[:len(dog_mask)//2]

# valid_img = cat_img[len(cat_img)//2:] + dog_img[len(dog_img)//2:]
# valid_mask = cat_mask[len(cat_mask)//2:] + dog_mask[len(dog_mask)//2:]


# ########################################################################
# ###########################     TEST      ##############################
# if train_img[500].split("/")[-1].split(".")[0] != train_mask[500].split("/")[-1].split(".")[0]:
#     print(f"img: {train_img[500]}")
#     print(f"mask: {train_mask[500]}")
#     raise AssertionError("EQUALITY TEST: FAILED (TRAIN)")
# else:
#     print(f"img: {train_img[500]}")
#     print(f"mask: {train_mask[500]}")
#     print("EQUALITY TEST: PASSED (TRAIN)")

# if valid_img[500].split("/")[-1].split(".")[0] != valid_mask[500].split("/")[-1].split(".")[0]:
#     print(f"img: {valid_img[500]}")
#     print(f"mask: {valid_mask[500]}")
#     raise AssertionError("EQUALITY TEST: FAILED (VALIDATION)")
# else:
#     print(f"img: {valid_img[500]}")
#     print(f"mask: {valid_mask[500]}")
#     print("EQUALITY TEST: PASSED (VALIDATION)")
# ########################################################################


# train_img_dict = {img_dir : train_img}
# train_mask_dict = {mask_dir : train_mask}

# valid_img_dict = {img_dir : valid_img}
# valid_mask_dict = {mask_dir : valid_mask}


# train_dataset = Load_Dataset(img_dir = train_img_dict,  # either str of directory or list of directory
#                        mask_dir = train_mask_dict, # either str of directory or list of directory
#                        specie_dict = specie_dict,  
#                        classes=["background","cat","dog"], 
#                        classes_limit=False, 
#                        augmentation=False, # insert augmentation function here
#                        preprocessing=preprocess_fn,# insert preprocessing function here
#                        fix_mask=True)

# valid_dataset = Load_Dataset(img_dir = valid_img_dict,  # either str of directory or list of directory
#                        mask_dir = valid_mask_dict, # either str of directory or list of directory
#                        specie_dict = specie_dict,  
#                        classes=["background","cat","dog"], 
#                        classes_limit=False, 
#                        augmentation=False, # insert augmentation function here
#                        preprocessing=preprocess_fn,# insert preprocessing function here
#                        fix_mask=True)



# # Batch Gradient Descent: Batch Size = Size of Training Set
# # Stochastic Gradient Descent: Batch Size = 1
# # Mini-Batch Gradient Descent: 1 < Batch Size < Size of Training Set

# loaded_train_dataset = Dataloader(dataset = train_dataset,
#                             dataset_size = len(train_dataset.img_path),
#                             batch_size = 29,
#                             shuffle = True
#                             )

# loaded_valid_dataset = Dataloader(dataset = valid_dataset,
#                             dataset_size = len(valid_dataset.img_path),
#                             batch_size = 29,
#                             shuffle = False
#                             )