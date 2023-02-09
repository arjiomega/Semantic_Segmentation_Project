
# from package import data
from config import config
import random
import pytest
import warnings
import pandas as pd
from pathlib import Path
import copy
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
            if method.__name__ in ["test_CleanData_duplicate","test_splitdata"]:
                _,_ = self.test_Clean.remove_unlabeled_data()

                # removing duplicates take a lot of time so it is excluded from test_splitdata
                # if method.__name__ in ["test_splitdata"]:
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

            if ds == "train":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (1659+3484)
            elif ds == "valid":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (474+995)
            elif ds == "test":
                assert len(temp_img_list) == len(temp_mask_list)
                assert len(temp_img_list) == (237+497+3)

