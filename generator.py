import cv2
import numpy as np
import glob

batch_size =4
train_dir = '/content/drive/My Drive/Colab/U-Net/Figaro1k/train'
val_dir = '/content/drive/My Drive/Colab/U-Net/Figaro1k/val'

train_filenames = glob.glob(train_dir+"/Images/*.jpg")
val_filenames = glob.glob(val_dir+"/Images/*.jpg")

input_size = 256

# def train_generator():
#     x = []
#     y = []
#     for image_filename in train_filenames:
#         print (image_filename)
#         img  = cv2.imread(image_filename)
#         img  = cv2.resize(img, (input_size, input_size))
#         mask_filename = image_filename.replace("Images", "Masks").replace("org.jpg", "gt.pbm")
#         mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, (input_size, input_size))
#         x.append(img)
#         y.append(mask)
#     x = np.array(x, np.float32) / 255
#     y = np.array(y, np.float32) / 255
#     return x, y

def train_generator(batch_size=batch_size):
    x = []
    y = []
    for image_filename in train_filenames:
        print (image_filename)
        img  = cv2.imread(image_filename)
        img  = cv2.resize(img, (input_size, input_size))
        mask_filename = image_filename.replace("Images", "Masks").replace("org.jpg", "gt.pbm")
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (input_size, input_size))
        x.append(img)
        y.append(mask)
        if len(x) == batch_size:
            imgs = np.array(x, np.float32) / 255
            masks = np.array(y, np.float32) / 255
            x = []
            y = []
            yield imgs, masks




# def val_generator():
#     x = []
#     y = []
#     for image_filename in val_filenames:
#         img  = cv2.imread(image_filename)
#         img  = cv2.resize(img, (input_size, input_size))
#         mask_filename = image_filename.replace("Images", "Masks").replace("org.jpg", "gt.pbm")
#         mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, (input_size, input_size))
#         x.append(img)
#         y.append(mask)
#     x = np.array(x, np.float32) / 255
#     y = np.array(y, np.float32) / 255
#     return x, y

def val_generator(batch_size=batch_size):
    x = []
    y = []
    for image_filename in val_filenames:
        img  = cv2.imread(image_filename)
        img  = cv2.resize(img, (input_size, input_size))
        mask_filename = image_filename.replace("Images", "Masks").replace("org.jpg", "gt.pbm")
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (input_size, input_size))
        x.append(img)
        y.append(mask)
        if len(x) == batch_size:
            imgs = np.array(x, np.float32) / 255
            masks = np.array(y, np.float32) / 255
            x = []
            y = []
            yield imgs, masks

if __name__ == '__main__':
    train_generator(batch_size=batch_size)
    val_generator(batch_size=batch_size)