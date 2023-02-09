
from package import utils
import data_utilities
from package import load_data



def train():
    """train model on dataset
    """

    utils.set_seeds()

    train_dataset, valid_dataset = load_data()

    loaded_train_dataset = data_utilities.Dataloader(dataset = train_dataset,
                            dataset_size = len(train_dataset.img_path),
                            batch_size = 29,
                            shuffle = True
                            )

    loaded_valid_dataset = data_utilities.Dataloader(dataset = valid_dataset,
                                dataset_size = len(valid_dataset.img_path),
                                batch_size = 29,
                                shuffle = False
                                )



    # Model (Temporary)
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
