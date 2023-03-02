
from package import utils

from package.data import load_data
from package.data_utilities import dataloader

import tensorflow as tf
import segmentation_models as sm
import numpy as np
import keras

def train(args,trial_run=True):
    """train model on dataset
    args:
        "encoder": "vgg16",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 40,
        "loss_function": "combined_focal_dice_loss"
    """

    utils.set_seeds()

    train_dataset, valid_dataset, test_dataset = load_data()

    # # Batch Gradient Descent: Batch Size = Size of Training Set
    # # Stochastic Gradient Descent: Batch Size = 1
    # # Mini-Batch Gradient Descent: 1 < Batch Size < Size of Training Set
    # loaded_train_dataset = dataloader.Dataloader(dataset = train_dataset,
    #                         dataset_size = len(train_dataset.img_path),
    #                         batch_size = 29,
    #                         shuffle = True
    #                         )

    # loaded_valid_dataset = dataloader.Dataloader(dataset = valid_dataset,
    #                             dataset_size = len(valid_dataset.img_path),
    #                             batch_size = 29,
    #                             shuffle = False
    #                             )

    loaded_test_dataset = dataloader.Dataloader(dataset = test_dataset,
                                dataset_size = len(test_dataset.img_path),
                                batch_size = 29,
                                shuffle = False
                                )

    if trial_run == False:
        # Model (Temporary)
        model = sm.Unet(args.encoder, args.n_classes, activation='softmax')

        optim = tf.keras.optimizers.Adam(args.learning_rate)

        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 2, 2]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        model.compile(optim,total_loss,metrics)

        # Training
        callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
        ]

        model_history = model.fit(loaded_train_dataset,
                                  steps_per_epoch = len(loaded_train_dataset),
                                  epochs = args.epochs,
                                  callbacks = callbacks,
                                  validation_data = loaded_valid_dataset,
                                  validation_steps = len(loaded_valid_dataset),
                                    )

    else:
        model = sm.Unet('vgg16', classes=3, activation='softmax')
        optim = tf.keras.optimizers.Adam(0.0001)
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 2, 2]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        model.compile(optim,total_loss,metrics)

        # Load the weights saved using the ModelCheckpoint callback
        model.load_weights('./best_model.h5')

    # performance on test dataset
    performance = model.evaluate(loaded_test_dataset)


    return {
        'args':args,
        'model':model,
        'performance':performance,
    }

if __name__ == "__main__":
    train()