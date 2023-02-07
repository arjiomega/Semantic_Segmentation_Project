import os
import pandas as pd
import numpy as np
import math
from config import config

import data_utilities

data_utilities.Clean_Data
data_utilities.Load_Dataset
data_utilities.Dataloader

config.IMG_DIR
config.MASK_DIR

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

    # add this to test
    assert len(img_inputs) == len(mask_inputs), "img_inputs and mask_inputs do not have the same length"

    return img_inputs, mask_inputs


def split_data(img_inputs,mask_inputs, include_test=False):
    # divide equally to prevent class imbalance in dataset
    ## this one may produce errors again

    # change 2370 by finding the index of last cat in the sorted list input
    cat_img = img_inputs[:2370]
    dog_img = img_inputs[2370:]

    cat_mask = mask_inputs[:2370]
    dog_mask = mask_inputs[2370:]


    train_img = cat_img[:len(cat_img)//2] + dog_img[:len(dog_img)//2]
    train_mask = cat_mask[:len(cat_mask)//2] + dog_mask[:len(dog_mask)//2]

    valid_img = cat_img[len(cat_img)//2:] + dog_img[len(dog_img)//2:]
    valid_mask = cat_mask[len(cat_mask)//2:] + dog_mask[len(dog_mask)//2:]

    train_img_dict = {img_dir : train_img}
    train_mask_dict = {mask_dir : train_mask}

    valid_img_dict = {img_dir : valid_img}
    valid_mask_dict = {mask_dir : valid_mask}

    train = {"img": , "mask":}
    valid = {"img": , "mask":}

    datasets = {"train":{"img": 10, "mask": 10},
                "valid":{"img": 10, "mask": 10} }



def preprocess():
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
    clean_data = data_utilities.Clean_Data(img_inputs = img_inputs,
                                           mask_inputs = mask_inputs,
                                           specie_dict = specie_dict)

    img_inputs,mask_inputs = clean_data.img_ids, clean_data.mask_ids

    # Split Data
    """
    Each number here are images and are replaced by their specie index
    [1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    split into train, valid, test with equal number of classes
    use for loop and get 1 from each classes until we reach the desired amount (Ex: 0.7x(Tot_num_samples) )
    If reached desired amount continue to valid then test
    USE:
        1. img_inputs
        2. mask_inputs
        3. specie_dict
    MUST:
        1. img_inputs and mask_inputs name must be similar (Ex: "dog_name01.jpg" = "dog_name01.png")
    PROCESS:
        1. get a list of unique classes (Ex: From [1,1,1,2,2,2,3,4] to [1,2,3,4])
        2. get a random image and mask from each class until desired amount per dataset
    """

    train_percent = 0.7
    valid_percent = 0.12
    test_percent = 0.10




    specie_list = [specie_dict[img_] for img_ in img_inputs]

    # 0: background, 1: cat, 2: dog
    cat_img = [img_ for img_ in img_inputs if specie_dict[img_] == 1]
    dog_img = [img_ for img_ in img_inputs if specie_dict[img_] == 2]


    split_setting = {"train" : 0.7,
                     "valid" : 0.2,
                     "test" : 0.1
    }


    all_dataset = {}

    for ds in ["train","valid"]:
        img_list = []
        mask_list = []

        dataset_length = math.floor( split_setting[ds] * len(img_inputs))

        for specie_index in np.unique( list( specie_dict.values() ) ):
        # get a file_name from img_inputs and mask_inputs using specie_index and specie_dict

            # temp_img_list and temp_mask_list for each specie (prevent adding to list if it reaches desired dataset_length)
            temp_img_list =  [img_ for img_ in img_inputs    for counter in range(dataset_length)   if specie_dict[{}.format(img_.split(".")[0])]  == specie_index and counter < dataset_length]
            temp_mask_list = [mask_ for mask_ in mask_inputs for counter in range(dataset_length)   if specie_dict[{}.format(mask_.split(".")[0])] == specie_index and counter < dataset_length]

            # extend to combine all lists with different specie
            img_list.extend(temp_img_list)
            mask_list.extend(temp_mask_list)


        all_dataset[ds: {"img" : img_list,
                         "mask": mask_list}]




    out_datasets = {"train":{"img": temp_img_list, "mask": temp_mask_list} }








    cat_img = img_inputs[:2370]
    dog_img = img_inputs[2370:]

    cat_mask = mask_inputs[:2370]
    dog_mask = mask_inputs[2370:]


    train_img = cat_img[:len(cat_img)//2] + dog_img[:len(dog_img)//2]
    train_mask = cat_mask[:len(cat_mask)//2] + dog_mask[:len(dog_mask)//2]

    valid_img = cat_img[len(cat_img)//2:] + dog_img[len(dog_img)//2:]
    valid_mask = cat_mask[len(cat_mask)//2:] + dog_mask[len(dog_mask)//2:]

    train_img_dict = {img_dir : train_img}
    train_mask_dict = {mask_dir : train_mask}

    valid_img_dict = {img_dir : valid_img}
    valid_mask_dict = {mask_dir : valid_mask}

    datasets = {"train":{"img": 10, "mask": 10},
                "valid":{"img": 10, "mask": 10} }





    train_dataset = data_utilities.Load_Dataset(img_dir = train_img_dict,  # either str of directory or list of directory
                        mask_dir = train_mask_dict, # either str of directory or list of directory
                        specie_dict = specie_dict,
                        classes=["background","cat","dog"],
                        classes_limit=False,
                        augmentation=False, # insert augmentation function here
                        preprocessing=preprocess_fn,# insert preprocessing function here
                        fix_mask=True)

    valid_dataset = data_utilities.Load_Dataset(img_dir = valid_img_dict,  # either str of directory or list of directory
                        mask_dir = valid_mask_dict, # either str of directory or list of directory
                        specie_dict = specie_dict,
                        classes=["background","cat","dog"],
                        classes_limit=False,
                        augmentation=False, # insert augmentation function here
                        preprocessing=preprocess_fn,# insert preprocessing function here
                        fix_mask=True)

    return 0





def preprocess_fn(img,mask):

    # temporary
    preprocessed_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    preprocessed_mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)

    return preprocessed_img, preprocessed_mask








"""NOTE:
    1. Consider splitting classes equally
    2. img_dir and img_name in pandas dataframe, try to make then equal in each element
"""

img_dir = "images"
mask_dir = "annotations/trimaps"

# remove hidden files and not jpg or png
img_inputs = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
mask_inputs = [mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
img_inputs = sorted(img_inputs)
mask_inputs = sorted(mask_inputs)

assert len(img_inputs) == len(mask_inputs), "img_inputs and mask_inputs do not have the same length"

# divide equally to prevent class imbalance in dataset
## this one may produce errors again 
cat_img = img_inputs[:2370]
dog_img = img_inputs[2370:]

cat_mask = mask_inputs[:2370]
dog_mask = mask_inputs[2370:]


train_img = cat_img[:len(cat_img)//2] + dog_img[:len(dog_img)//2]
train_mask = cat_mask[:len(cat_mask)//2] + dog_mask[:len(dog_mask)//2]

valid_img = cat_img[len(cat_img)//2:] + dog_img[len(dog_img)//2:]
valid_mask = cat_mask[len(cat_mask)//2:] + dog_mask[len(dog_mask)//2:]


########################################################################
###########################     TEST      ##############################
if train_img[500].split("/")[-1].split(".")[0] != train_mask[500].split("/")[-1].split(".")[0]:
    print(f"img: {train_img[500]}")
    print(f"mask: {train_mask[500]}")
    raise AssertionError("EQUALITY TEST: FAILED (TRAIN)")
else:
    print(f"img: {train_img[500]}")
    print(f"mask: {train_mask[500]}")
    print("EQUALITY TEST: PASSED (TRAIN)")

if valid_img[500].split("/")[-1].split(".")[0] != valid_mask[500].split("/")[-1].split(".")[0]:
    print(f"img: {valid_img[500]}")
    print(f"mask: {valid_mask[500]}")
    raise AssertionError("EQUALITY TEST: FAILED (VALIDATION)")
else:
    print(f"img: {valid_img[500]}")
    print(f"mask: {valid_mask[500]}")
    print("EQUALITY TEST: PASSED (VALIDATION)")
########################################################################


train_img_dict = {img_dir : train_img}
train_mask_dict = {mask_dir : train_mask}

valid_img_dict = {img_dir : valid_img}
valid_mask_dict = {mask_dir : valid_mask}


train_dataset = Load_Dataset(img_dir = train_img_dict,  # either str of directory or list of directory
                       mask_dir = train_mask_dict, # either str of directory or list of directory
                       specie_dict = specie_dict,  
                       classes=["background","cat","dog"], 
                       classes_limit=False, 
                       augmentation=False, # insert augmentation function here
                       preprocessing=preprocess_fn,# insert preprocessing function here
                       fix_mask=True)

valid_dataset = Load_Dataset(img_dir = valid_img_dict,  # either str of directory or list of directory
                       mask_dir = valid_mask_dict, # either str of directory or list of directory
                       specie_dict = specie_dict,  
                       classes=["background","cat","dog"], 
                       classes_limit=False, 
                       augmentation=False, # insert augmentation function here
                       preprocessing=preprocess_fn,# insert preprocessing function here
                       fix_mask=True)



# Batch Gradient Descent: Batch Size = Size of Training Set
# Stochastic Gradient Descent: Batch Size = 1
# Mini-Batch Gradient Descent: 1 < Batch Size < Size of Training Set

loaded_train_dataset = Dataloader(dataset = train_dataset,
                            dataset_size = len(train_dataset.img_path),
                            batch_size = 29,
                            shuffle = True
                            )

loaded_valid_dataset = Dataloader(dataset = valid_dataset,
                            dataset_size = len(valid_dataset.img_path),
                            batch_size = 29,
                            shuffle = False
                            )