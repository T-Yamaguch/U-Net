from model import Unet
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

checkpoint_path = "checkpoint/cp.ckpt"
input_shape=(256,256,3)
input_size = 256
val_dir = 'Figaro1k/val'
val_filenames = glob.glob(val_dir+"/Images/*.jpg")
threshold=0.5

model = Unet(input_shape=input_shape)
model.load_weights(filepath=checkpoint_path)


def generator():
    while 1:
        x = []
        for image_filename in val_filenames:
            img  = cv2.imread(image_filename)
            (img_H, img_W, img_CH) = img.shape
            img_resize  = cv2.resize(img, (input_size, input_size))
            x.append(img_resize)
            x= np.array(x, np.float32) / 255
            pred = model.predict(x, steps=1)
            prob = np.array(pred > threshold).astype(np.float32)*255
            prob=prob.reshape(input_size, input_size, 1)
            prob  = cv2.resize(prob, ( img_W, img_H))
            cv2.imwrite('temp.jpg', prob)
            mask = cv2.imread('temp.jpg')

            img_sync = cv2.addWeighted(img, 0.5,mask, 0.5, 0)
            cv2.imwrite('temp2.jpg', img_sync)

            result  = Image.open('temp2.jpg')
            plt.imshow(result)
            plt.pause(0.1)

            x = []

if __name__ == '__main__':
    generator()
