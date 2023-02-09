
import imagehash
from PIL import Image
from pathlib import Path
from config import config

class Clean_Data:
    def __init__(
                 self,
                 img_inputs,
                 mask_inputs,
                 specie_dict,
                ):

        #print("\nRunning: Clean_Data")
        #print("------------------------------------\n")

        self.specie_dict = specie_dict
        self.img_ids = img_inputs
        self.mask_ids = mask_inputs

        # # TEST
        # ## CHECK LENGTH
        # if len(self.img_ids) != len(self.mask_ids):
        #     raise AssertionError("Lists of Id and path do not have the same length")
        # else:
        #     print("Length Check Passed!\n")

        self.count_remove = 0
        #self.run_clean()

        #print("\nStop: Clean_Data")
        #print("------------------------------------\n")


    def run_clean(self):
        # remove in the list those without labels

        print("Removing Unlabeled Data")
        self.img_ids, self.mask_ids = self.remove_unlabeled_data()
        print(f"Number of data removed: {self.count_remove}")

        print("Removing Duplicate Data")
        self.img_ids, self.mask_ids = self.remove_duplicate_data()
        print(f"Number of data removed: {self.count_remove}")


    def remove_unlabeled_data(self):

        # name without filetype .jpeg
        name_list = [img_id.split('.')[0] for img_id in self.img_ids]

        # list of img_name without label
        no_label_list = [img_name for img_name in name_list if img_name not in self.specie_dict]

        self.count_remove += len(no_label_list)

        temp_img = self.img_ids
        temp_mask = self.mask_ids

        self.img_ids = [id for index,id in enumerate(temp_img) if name_list[index] not in no_label_list]
        self.mask_ids = [id for index,id in enumerate(temp_mask) if name_list[index] not in no_label_list]

        return self.img_ids, self.mask_ids


    def remove_duplicate_data(self):
        img_hashes = {}
        duplicate_list = []

        # get a list of duplicate images
        for img_id in self.img_ids:
            img_path = Path(config.IMG_DIR,img_id)

            hash = imagehash.average_hash(Image.open(img_path))

            if hash in img_hashes:
                duplicate_list.append(img_id)
            else:
                img_hashes[hash] = img_id

        self.count_remove += len(duplicate_list)

        temp_img_ids = self.img_ids
        temp_mask_ids = self.mask_ids

        self.img_ids =  [path for index,path in enumerate(temp_img_ids) if temp_img_ids[index] not in duplicate_list]
        self.mask_ids = [path for index,path in enumerate(temp_mask_ids) if temp_img_ids[index] not in duplicate_list]

        return self.img_ids, self.mask_ids


