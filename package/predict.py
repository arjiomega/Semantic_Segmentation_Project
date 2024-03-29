from pathlib import Path


import cv2
import numpy as np
import tensorflow as tf

from config import config
from loss_functions import DiceLoss

cap = cv2.VideoCapture(str(Path(config.DATA_DIR,"dog-petting-cat-pet.mp4")))

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

custom_objects = {'DiceLoss': DiceLoss}

model_file = "model_run_subclass_split_dice_continue_train.h5"

loaded_model = tf.keras.models.load_model(str(Path(config.DATA_DIR,model_file)), custom_objects=custom_objects, compile=False)



while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # reset frame if we reach total number of frames
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success,frame = cap.read()

    preprocessed_img = cv2.resize(frame, (128,128), interpolation=cv2.INTER_LINEAR)
    preprocessed_img = np.round(preprocessed_img)

    preprocessed_img = preprocessed_img/255

    before = preprocessed_img
    preprocessed_img = np.expand_dims(preprocessed_img, axis = 0)
    predict_mask = loaded_model.predict(preprocessed_img)

    threshold = 0.9
    predict_mask = np.where(predict_mask > threshold, 1, 0)

    predict_mask = predict_mask.astype(np.uint8)

    _,width,height,n_classes = predict_mask.shape

    print(n_classes)

    # bg, cat, dog
    color_list = [[255,0,0],[0,255,0],[0,0,255]]

    temp = before

    ###
    test = temp
    contours,hierarcy = cv2.findContours(predict_mask[...,1].squeeze(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    testing = cv2.drawContours(test,contours,-1,(0,255,0),3)
    ###

    for i in range(1,n_classes):

        final_mask = np.where(predict_mask[...,i].squeeze()[...,None],color_list[i],before)

        temp = cv2.addWeighted(temp,0.9,final_mask,0.1,0)


    cv2.imshow('name',temp)



    c = cv2.waitKey(1)

    if c & 0xFF == ord('q'):
        break
