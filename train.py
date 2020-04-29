from model import Unet
from generator import train_generator, val_generator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import glob
import numpy as np

train_dir = '/content/drive/My Drive/Colab/U-Net/Figaro1k/train'
val_dir = '/content/drive/My Drive/Colab/U-Net/Figaro1k/val'

train_filenames = glob.glob(train_dir+"/Images/*.jpg")
val_filenames = glob.glob(val_dir+"/Images/*.jpg")

input_shape=(256,256,3)

activation= "sigmoid"
loss= "binary_crossentropy"
optimizer= "adam"
learning_rate= 1e-4
epochs= 100
batch_size= 4
threshold= 0.7

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


def main():
    model = Unet(input_shape=input_shape)
    model.summary()
                                 
    model.fit_generator(generator=train_generator(batch_size=batch_size),
                        steps_per_epoch=np.ceil(float(len(train_filenames)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=TensorBoard(log_dir='logs'),
                        validation_data=val_generator(batch_size=batch_size),
                        validation_steps=np.ceil(float(len(val_filenames)) / float(batch_size)),
                        )

if __name__ == '__main__':
    main()