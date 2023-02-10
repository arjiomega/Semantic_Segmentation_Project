
# from package import data
from config import config
import random
import pytest
import warnings
import pandas as pd
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
with warnings.catch_warnings():
    # ignore warning on tensorflow (package/data_utilities/dataloader)
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")
    from package import data
    from package.data_utilities import clean_data,load_dataset,dataloader

import sys
import math

class TestData:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""

        df = pd.read_csv(Path(config.MASK_DIR.parent,"list.txt"), comment='#',header=None, sep=" ")
        df.columns = ['image_name','class_index','specie_index','breed_index']
        df = df.sort_values(by = 'image_name', ascending=True)
        df = df.reset_index(drop=True)

        # Get dict ("image_name": specie_index)
        cls.specie_dict = df.set_index("image_name")["specie_index"].to_dict()


    @classmethod
    def teardown_class(cls):
        """Called after every class initialization."""
        pass

    def setup_method(self,method):
        """Called before every method."""
        #self.label_encoder = data.LabelEncoder()

        if method.__name__  not in ["test_specie","test_data2list"]:  #in ["test_CleanData_unlabeled","test_CleanData_duplicate","test_splitdata"]:
            self.test_img,self.test_mask = data.data2list(config.IMG_DIR, config.MASK_DIR)
            self.test_Clean = clean_data.Clean_Data(img_inputs=self.test_img,
                                           mask_inputs=self.test_mask,
                                           specie_dict=self.specie_dict
            )
            if method.__name__ in ["test_CleanData_duplicate","test_splitdata","test_loadDataset"]:
                _,_ = self.test_Clean.remove_unlabeled_data()

                # removing duplicates take a lot of time so it is excluded from test_splitdata
                # can also be excluded in duplicates because we can still load them and do test
                # if method.__name__ in ["test_loadDataset"]:
                #     _,_ = self.test_Clean.remove_duplicate_data()


    def teardown_method(self,method):
        """Called after every method."""
        if method.__name__ in ["test_CleanData_unlabeled","test_CleanData_duplicate","test_splitdata"]:
            del self.test_img,self.test_mask
            del self.test_Clean


    def test_specie(self):

        #test if cat
        assert self.specie_dict["Sphynx_183"] == 1
        # test if dog
        assert self.specie_dict["american_bulldog_100"] == 2

    def test_data2list(self):

        test_img,test_mask = data.data2list(config.IMG_DIR, config.MASK_DIR)

        # test length
        assert len(test_img) == len(test_mask)

        # test on random positions
        for i in random.sample(range(0,len(test_img)),10):
            assert test_img[i].split(".")[0] == test_mask[i].split(".")[0]

    # Test Clean Data Class
    def test_CleanData_unlabeled(self):
        img_list,mask_list = self.test_Clean.remove_unlabeled_data()
        assert len(img_list) == len(mask_list)
        for i in random.sample(range(0,len(img_list)),10):
            assert img_list[i].split(".")[0] == mask_list[i].split(".")[0]

    def test_CleanData_duplicate(self):
        img_list,mask_list = self.test_Clean.remove_duplicate_data()
        assert len(img_list) == len(mask_list)
        for i in random.sample(range(0,len(img_list)),10):
            assert img_list[i].split(".")[0] == mask_list[i].split(".")[0]

    def test_splitdata(self):
        img_inputs,mask_inputs = self.test_Clean.img_ids, self.test_Clean.mask_ids
        specie_dict = self.specie_dict

        ClassList_dict = {}
        classes = ["background","cat","dog"]


        for class_ in classes:
            specie_index = classes.index(class_)
            ClassList_dict[class_] = {"img" : [img_  for img_  in img_inputs  if specie_dict[(img_.split(".")[0])]  == specie_index],
                                        "mask": [mask_ for mask_ in mask_inputs if specie_dict[(mask_.split(".")[0])] == specie_index]
                                        }

        split_setting = {"train" : 0.7,
                        "valid" : 0.2,
                        "test" : 0.1
                        }
        include_test = True
        if not include_test:
            split_setting["valid"] = split_setting["test"]
            del split_setting["test"]

        dataset_list = list(split_setting.keys())

        all_dataset = {}

        assert len(ClassList_dict["background"]["img"]) == 0
        assert len(ClassList_dict["cat"]["img"]) == 2371
        assert len(ClassList_dict["dog"]["img"]) == 4978
        assert dataset_list == ["train","valid","test"]

        prev_length = [0] * len(classes)
        start = [0] * len(classes)
        end = [0] * len(classes)

        for ds in dataset_list:
            temp_img_list = []
            temp_mask_list = []

            for class_ in classes:
                if len(ClassList_dict[class_]["img"]) == 0:
                    continue
                assert class_ != "background"

                specie_index = classes.index(class_)

                start = copy.deepcopy(prev_length)

                dataset_length = math.floor( split_setting[ds] * len(ClassList_dict[class_]["img"]))

                end[specie_index] = start[specie_index] + dataset_length


                print(f"ds: {ds} || class: {class_}")
                print(f"start: {start[specie_index]} || dataset_length: {dataset_length} || end: {end[specie_index]}")

                if ds == "train":
                    if class_ == "cat":
                        assert start[specie_index] == 0
                        assert dataset_length == 1659
                        assert end[specie_index] == 1659
                    elif class_ == "dog":
                        assert start[specie_index] == 0
                        assert dataset_length == 3484
                        assert end[specie_index] == 3484

                elif ds == "valid":
                    if class_ == "cat":
                        assert start[specie_index] == 1659
                        assert dataset_length == 474
                        assert end[specie_index] == (1659+474)
                    elif class_ == "dog":
                        assert start[specie_index] == 3484
                        assert dataset_length == 995
                        assert end[specie_index] == (3484+995)

                elif ds == "test":
                    if class_ == "cat":
                        assert start[specie_index] == (1659+474)
                        assert dataset_length == 237
                        assert end[specie_index] == (1659+474+237)
                    elif class_ == "dog":
                        assert start[specie_index] == (3484+995)
                        assert dataset_length == 497
                        assert end[specie_index] == (3484+995+497)

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

            for i in random.sample(range(0,len(temp_img_list)),10):
                assert temp_img_list[i].split(".")[0] == temp_mask_list[i].split(".")[0]

            if ds == "train":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (1659+3484)
            elif ds == "valid":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (474+995)
            elif ds == "test":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (237+497+3)

        # test whole function
        img_inputs,mask_inputs = self.test_Clean.img_ids, self.test_Clean.mask_ids
        specie_dict = self.specie_dict
        test_all_dataset = data.split_data(img_inputs,mask_inputs,specie_dict)

        for ds in dataset_list:
            temp_img_list = test_all_dataset[ds]["img"]
            temp_mask_list = test_all_dataset[ds]["mask"]
            if ds == "train":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (1659+3484)
            elif ds == "valid":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (474+995)
            elif ds == "test":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (237+497+3)

    def test_loadDataset(self):
        # removed unlabeled data
        img_inputs,mask_inputs = self.test_Clean.img_ids, self.test_Clean.mask_ids
        specie_dict = self.specie_dict

        # split
        all_dataset = data.split_data(img_inputs,mask_inputs,specie_dict)

        # get train
        train_img = all_dataset["train"]["img"]
        train_mask = all_dataset["train"]["mask"]

        assert train_img == sorted(train_img)
        assert train_mask == sorted(train_mask)

        img_path = [str(Path(config.IMG_DIR,img_id)) for img_id in train_img]
        mask_path = [str(Path(config.MASK_DIR,mask_id)) for mask_id in train_mask]

        assert img_path == sorted(img_path)
        assert mask_path == sorted(mask_path)
        assert len(img_path) == len(mask_path)

        for i in random.sample(range(0,len(img_path)),10):
            assert img_path[i].split("/")[-1].split(".")[0] == train_img[i].split(".")[0]
            assert img_path[i].split("/")[-1].split(".")[0] == mask_path[i].split("/")[-1].split(".")[0]

        classes_limit = False
        classes = ["background", "cat", "dog"]
        if classes_limit:
            class_val = [classes.index(cls) for cls in classes_limit]
        else:
            class_val = [i for i,x in enumerate(classes)]

        ## TEST CAT
        mask = cv2.imread(mask_path[10],0)
        specie_index = specie_dict.get(mask_path[10].split("/")[-1].split(".")[0])

        assert all(np.unique(mask) == [1,2,3])

        bg = 0
        condition_object = np.logical_or(mask == 1, mask ==3)
        condition_bg = (mask == 2)

        mask = np.where(condition_object,specie_index,np.where(condition_bg,bg,mask))

        assert all(np.unique(mask) == [0,1])

        masks = [(mask == class_) for class_ in class_val]
        mask = np.stack(masks, axis = -1).astype('float')

        # mask channel is the number of classes
        assert mask.shape[-1] == len(class_val)

        print(np.unique(mask[...,0]))
        print(np.unique(mask[...,1]))
        print(np.unique(mask[...,2]))
        assert len(np.unique(mask[...,0])) == 2 # background exists
        assert len(np.unique(mask[...,1])) == 2 # cat exists
        assert len(np.unique(mask[...,2])) == 1 # dog does not exists

        # PREPROCESS
        preprocessed_mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)
        preprocessed_mask = np.round(preprocessed_mask)

        assert len(np.unique(preprocessed_mask)) == 2

        # run complete
        train_dataset = load_dataset.Load_Dataset(img_dir = config.IMG_DIR,
                                                mask_dir = config.MASK_DIR,
                                                img_list = train_img,
                                                mask_list= train_mask,
                                                specie_dict = specie_dict,
                                                classes=["background","cat","dog"],
                                                classes_limit=False,
                                                augmentation=False, #augmentation function
                                                preprocessing=data.preprocess_fn,#preprocessing function
                                                fix_mask=True)
        #img,mask = test_dataset[5]
        assert train_dataset.class_count["cat"] == 1659
        assert train_dataset.class_count["dog"] == 3484

        # DOG TEST
        dog_test_img,dog_test_mask = train_dataset[3000]

        assert train_dataset.img_ids[500].split(".")[0] == train_dataset.mask_ids[500].split(".")[0]
        # 1: does not exist || 2: does exist (in the image)
        assert len(np.unique(dog_test_mask[...,0])) == 2 # background
        assert len(np.unique(dog_test_mask[...,1])) == 1 # cat
        assert len(np.unique(dog_test_mask[...,2])) == 2 # dog

        # CAT TEST
        cat_test_img,cat_test_mask = train_dataset[3000]

        # 1: does not exist || 2: does exist (in the image)
        assert len(np.unique(cat_test_mask[...,0])) == 2 # background
        assert len(np.unique(cat_test_mask[...,1])) == 1 # cat
        assert len(np.unique(cat_test_mask[...,2])) == 2 # dog


    def test_final(self):
        train_dataset, valid_dataset, test_dataset = data.load_data(run_pytest=True)

        # DOG TEST
        dog_test_img,dog_test_mask = train_dataset[3000]

        assert train_dataset.img_ids[500].split(".")[0] == train_dataset.mask_ids[500].split(".")[0]
        # 1: does not exist || 2: does exist (in the image)
        assert len(np.unique(dog_test_mask[...,0])) == 2 # background
        assert len(np.unique(dog_test_mask[...,1])) == 1 # cat
        assert len(np.unique(dog_test_mask[...,2])) == 2 # dog

        # CAT TEST
        cat_test_img,cat_test_mask = train_dataset[3000]

        # 1: does not exist || 2: does exist (in the image)
        assert len(np.unique(cat_test_mask[...,0])) == 2 # background
        assert len(np.unique(cat_test_mask[...,1])) == 1 # cat
        assert len(np.unique(cat_test_mask[...,2])) == 2 # dog
