

import os
import cv2
import numpy as np
from pathlib import Path


class Load_Dataset:
    def __init__(self,
                img_dir, mask_dir,
                img_list, mask_list,
                specie_dict,
                classes, classes_limit,
                augmentation, preprocessing, fix_mask):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = img_list
        self.mask_ids = mask_list

        self.specie_dict = specie_dict
        self.classes = classes

        self.img_path = [str(Path(self.img_dir,img_id)) for img_id in self.img_ids]
        self.mask_path = [str(Path(self.mask_dir,mask_id)) for mask_id in self.mask_ids]

        # Count classes and their number {"cat": 500, "dog": 1500}
        self.class_count = self.class_counter()

        # classes = 0: background || 1: cat || 2: dog   class_val = [0,1,2]
        if classes_limit:
            self.class_val = [self.classes.index(cls) for cls in classes_limit]
        else:
            self.class_val = [i for i,x in enumerate(self.classes)]

        self.fix_mask = fix_mask
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self,i):
        # read img and mask
        img = cv2.imread(self.img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path[i],0)

        if self.fix_mask:
            mask = self.fix(i,mask)

        # separate classes into their own channels
        ## if mask values in list of classes (class_values) return true
        masks = [(mask == class_) for class_ in self.class_val]
        ## stack 2d masks (list) into a single 3d array (True = 1, False = 0)
        ## axis = -1 > add another axis || example shape (500,500, n_classes)
        mask = np.stack(masks, axis = -1).astype('float')

        # apply augmentations
        if self.augmentation:
            img, mask = self.augmentation(img=img, mask=mask)

        # apply preprocessing
        if self.preprocessing:
            img, mask = self.preprocessing(img=img, mask=mask)

        # normalize image (0 -> 255) to (0 -> 1)
        img = img/255

        return img, mask


    def fix(self,i,mask):
        find_name = self.img_ids[i].split('.')[0]
        specie_index = self.specie_dict.get(self.mask_path[i].split("/")[-1].split(".")[0])

        if specie_index == None:
            print(f"{find_name} has no specie_index")

        bg = 0
        condition_object = np.logical_or(mask == 1, mask ==3)
        condition_bg = (mask == 2)
        mask = np.where(condition_object,specie_index,np.where(condition_bg,bg,mask))

        return mask

    def normalize(self,i,img,mask):
        return 0

    def class_counter(self):
        # classes=["background","cat","dog"]
        name_list = [class_.split("/")[-1].split(".")[0] for class_ in self.img_path]
        specie_list = [self.specie_dict.get(name,None) for name in name_list]
        class_count = {class_name: specie_list.count(class_id) for class_id,class_name in enumerate(self.classes) if class_name != "background"}

        return class_count