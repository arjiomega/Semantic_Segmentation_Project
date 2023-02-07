






class Load_Dataset:
    """NOTES:
        1. replace img_dir and mask_dir with list directly so i can create train and validation dataset (DONE)
        2. change img_path and mask_path because img_dir and mask_dir already cleaned before adding complete path (DONE)
        3. try to replace self.img_ids to os.listdir(img_dir) for direct cleaning to list (DONE)
        4. replace list into dictionary {"img_dir": ["dog_pic.jpg", ...]} (DONE)
        5. IMG and MASK are not the same again (DONE)
    """
    def __init__(self, img_dir, mask_dir, specie_dict,  classes, classes_limit, augmentation, preprocessing, fix_mask):

        print("\nRunning: Load_Dataset")
        print("------------------------------------\n")

        self.specie_dict = specie_dict
        self.classes = classes

        if isinstance(img_dir, str) and isinstance(mask_dir, str):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.img_ids = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
            self.mask_ids = [mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]

        elif isinstance(img_dir, dict) and isinstance(mask_dir, dict):
            self.img_dir, self.img_ids = list(img_dir.items())[0]
            self.mask_dir, self.mask_ids = list(mask_dir.items())[0]

        else:
            raise TypeError("Possible cause of error: \n1. img_dir and mask_dir are both not str or dict \n2. img_dir and mask_dir are neither str nor dict")

        self.img_ids = sorted(self.img_ids)
        self.mask_ids = sorted(self.mask_ids)

        if self.img_ids[500].split(".")[0] != self.mask_ids[500].split(".")[0]:
            print(f"img: {self.img_ids[500]}")
            print(f"mask: {self.img_ids[500]}")

            print("BEFORE")
            print(f"img: {self.img_ids[499]}")
            print(f"mask: {self.img_ids[499]}")
            print("AFTER")
            print(f"img: {self.img_ids[501]}")
            print(f"mask: {self.img_ids[501]}")

            raise AssertionError("img and mask are not the same")
        else:
            print("SORTED IDs PASSED!")


        self.img_path = [os.path.join(self.img_dir,img_id) for img_id in self.img_ids]
        self.mask_path = [os.path.join(self.mask_dir,mask_id) for mask_id in self.mask_ids]
        
        self.img_path = sorted(self.img_path)
        self.mask_path = sorted(self.mask_path)

        print("Lengths before data cleaning")
        print(f"img_ids len: {len(self.img_ids)}")
        print(f"mask_ids len: {len(self.mask_ids)}")
        print(f"img_path len: {len(self.img_path)}")
        print(f"mask_path len: {len(self.mask_path)}")

        if all(len(self.img_ids) != len(list_) for list_ in [self.mask_ids,self.img_path,self.mask_path] ):
            raise AssertionError("Lists of Id and path do not have the same length")
        else:
            print("Length Check Passed!\n")

        if self.img_path[500].split("/")[-1].split(".")[0] != self.mask_path[500].split("/")[-1].split(".")[0]:
            print(f"img: {self.img_path[500]}")
            print(f"mask: {self.mask_path[500]}")

            print("BEFORE")
            print(f"img: {self.img_path[499]}")
            print(f"mask: {self.mask_path[499]}")
            print("AFTER")
            print(f"img: {self.img_path[501]}")
            print(f"mask: {self.mask_path[501]}")

            raise AssertionError("img and mask are not the same")
        else:
            print("PASSED PATH COMPARISON BEFORE CLEANING")

        #################################
        #Data clean
        data_clean = Clean_Data(img_id_input = self.img_ids,
                                mask_id_input = self.mask_ids,
                                img_path_input = self.img_path,
                                mask_path_input = self.mask_path,
                                specie_dict_input = self.specie_dict)

        self.img_path = data_clean.img_paths
        self.mask_path = data_clean.mask_paths
        self.img_ids = data_clean.img_ids
        self.mask_ids = data_clean.mask_ids

        print("Lengths after data cleaning")
        print(f"img_ids len: {len(self.img_ids)}")
        print(f"mask_ids len: {len(self.mask_ids)}")
        print(f"img_path len: {len(self.img_path)}")
        print(f"mask_path len: {len(self.mask_path)}")

        if all(len(self.img_ids) != len(list_) for list_ in [self.mask_ids,self.img_path,self.mask_path] ):
            raise AssertionError("Lists of Id and path do not have the same length")
        else:
            print("Length Check Passed!\n")
        #################################

        # Count class dict {"cat": 500, "dog": 1500}
        self.class_count = self.class_counter()
        print(self.class_count,"\n")

        # classes = 0: background || 1: cat || 2: dog   class_val = [0,1,2]
        if classes_limit:
            self.class_val = [self.classes.index(cls) for cls in classes_limit]
        else:
            self.class_val = [i for i,x in enumerate(self.classes)]

        self.fix_mask = fix_mask

        self.augmentation = augmentation
        self.preprocessing = preprocessing


        name_list = [class_.split("/")[-1].split(".")[0] for class_ in self.img_path]
        mask_list = [class_.split("/")[-1].split(".")[0] for class_ in self.mask_path]

        # TWO TESTS BECAUSE FIRST ONE IS PRODUCING FALSE PASS RESULT!
        if all(x != y for x,y in zip(name_list,mask_list)):
            raise AssertionError("FAILED: img_path and mask_path are incorrectly paired!")
        else:
            print("PASSED FIRST TEST: img_path and mask_path are correctly paired!")

        compare_ = self.img_path[500].split("/")[-1].split(".")[0]
        compare_list = [ self.mask_path[500].split("/")[-1].split(".")[0],   ]

        if self.img_path[500].split("/")[-1].split(".")[0] != self.mask_path[500].split("/")[-1].split(".")[0]:
            print(f"img: {self.img_path[500]}")
            print(f"mask: {self.mask_path[500]}")
            raise AssertionError("FAILED: img_path and mask_path are incorrectly paired!")
        else:
            print("PASSED SECOND TEST: img_path and mask_path are correctly paired!")


        print("\nStop: Load_Dataset")
        print("------------------------------------\n")
    
    def __getitem__(self,i):

        if self.img_path[i].split("/")[-1].split(".")[0] != self.mask_path[i].split("/")[-1].split(".")[0]:
            print(f"img: {self.img_path[i]}")
            print(f"mask: {self.mask_path[i]}")
            raise AssertionError("img and mask are not the same")

        # read img and mask
        img = cv2.imread(self.img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path[i],0)

        if self.fix_mask:
            mask, specie_index = self.fix(i,mask)

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

        # fix values after preprocessing
        ## only works for the current problem (background, cat, dog)
        ## improve to work with multiple classes
        mask[mask > 0] = specie_index
        mask[mask <= 0] = 0

        return img, mask

    def fix(self,i,mask):
        find_name = self.img_ids[i].split('.')[0]
        #specie index 
        specie_index = self.specie_dict.get(find_name,None)
        if specie_index == None:
            print(f"{find_name} has no specie_index")


        #print("this is a test")
        #print(self.specie_dict.get("Egyptian_Mau_139",None))


        bg = 0
        condition_object = np.logical_or(mask == 1, mask ==3)
        condition_bg = (mask == 2)
        mask = np.where(condition_object,specie_index,np.where(condition_bg,bg,mask))

        return mask, specie_index
    
    def class_counter(self):

        # classes=["background","cat","dog"]
        name_list = [class_.split("/")[-1].split(".")[0] for class_ in self.img_path]
        specie_list = [self.specie_dict.get(name,None) for name in name_list] 
        class_count = {class_name: specie_list.count(class_id) for class_id,class_name in enumerate(self.classes) if class_name != "background"}


        return class_count