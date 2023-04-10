import random

def class_splitter(img_list:list[str], img_label:dict[str,int],file_ext:str) -> dict[str,list[str]]:
    """
    Split img_list into their own labels

    NOTE: if img_file from img_list does not exist in img_label, it will not be included in the output of this function

    Parameters:
    img_list:   List of images
    NOTE:       Remove file extension from each img_file
                Ex. ["dog_name.jpg",...] to ["dog_name",...]

    img_label:  Label for each img_file (Cat:1 , Dog:2)
    Ex.         ['dog_name':2,'cat_name':1]

    file_ext:   .jpg or .png after the file_name


    """
    class_split = {class_:[img+file_ext for img in img_list if img in img_label and img_label[img]==class_] for class_ in set(img_label.values())}

    return class_split

def dataset_splitter(img_list:list[str],mask_list:list[str],train:float,valid:float,test:float,shuffle:bool=False,rand_seed:int=5) -> dict[str, dict[str,list[str]]]:
    """
    Split data into train, valid, test datasets

    Parameters:
    img_list: list of images
    Ex. [dog_img.jpg,...]

    mask_list: list of masks
    Ex. [dog_mask.png,...]

    train,valid,test: fraction of data that is going to be split into train,valid,test sets

    rand_seed: to achieve similar results during shuffling
    """

    if shuffle:
        random.seed(rand_seed)

        zipped = list(zip(img_list,mask_list))
        random.shuffle(zipped)
        img_list, mask_list = zip(*zipped)


    assert round(train + valid + test,1) == 1.0, "sum of train,valid,test must be equal to 1"

    train_count = int(train*(len(img_list)))
    valid_count = int((train+valid)*(len(img_list)))
    test_count =  int((train+valid+test)*(len(img_list)))


    dataset = {"train":     {"img":img_list[:train_count],
                            "mask": mask_list[:train_count]},

               "valid":     {"img":img_list[train_count:valid_count],
                            "mask": mask_list[train_count:valid_count]},

               "test":      {"img":img_list[valid_count:test_count+1],
                            "mask": mask_list[valid_count:test_count+1]}

               }

    return dataset