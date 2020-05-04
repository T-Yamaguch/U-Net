from model import Unet
from generator import train_generator, val_generator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import glob
import numpy as np

import cv2

import time
import os

train_dir = 'Figaro1k/train'
val_dir = 'Figaro1k/val'

train_filenames = glob.glob(train_dir+"/Images/*.jpg")
val_filenames = glob.glob(val_dir+"/Images/*.jpg")

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

input_shape=(256,256,3)

epochs= 100
batch_size = 12


input_size = 256

def main():
    model = Unet(input_shape=input_shape)
    model.summary()
    model.fit_generator(generator=train_generator(batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(float(len(train_filenames)) / float(batch_size))),
                        epochs=epochs,
                        verbose=2,
#                        callbacks=[TensorBoard(log_dir='logs')],
                        callbacks=[TensorBoard(log_dir='logs'), 
                                   ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)],
                        validation_data=val_generator(batch_size=batch_size),
                        validation_steps=int(np.ceil(float(len(val_filenames)) / float(batch_size)))
                        )

if __name__ == '__main__':
    main()
