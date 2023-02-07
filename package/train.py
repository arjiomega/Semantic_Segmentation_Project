
from package import utils





def train():
    """train model on dataset
    """

    utils.set_seeds()

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


    # Model
    model = sm.Unet('vgg16', classes=3, activation='softmax')

    optim = tf.keras.optimizers.Adam(0.0001)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 2, 2])) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim,total_loss,metrics)

    # Training
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./drive/MyDrive/image_segmentation/test_best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
    ]

    model_history = model.fit(
    loaded_dataset,
    steps_per_epoch = len(loaded_dataset),
    epochs = 40,
    callbacks = callbacks,
    )
