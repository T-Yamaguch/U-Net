import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from tensorflow.keras.losses import binary_crossentropy

def Unet(input_shape=(256, 256, 3)):
    input_size = input_shape[0]

    inputs = Input(shape=input_shape)
    #256x256

    conv1 = Conv2D(32, (3, 3), padding = 'same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1_pool = MaxPooling2D((2,2), strides=(2,2))(conv1)
    #128x128

    conv2 = Conv2D(64, (3, 3), padding = 'same')(conv1_pool)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_pool = MaxPooling2D((2,2), strides=(2,2))(conv2)
    #64x64

    conv3 = Conv2D(128, (3, 3), padding = 'same')(conv2_pool)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3_pool = MaxPooling2D((2,2), strides=(2,2))(conv3)
    #32x32

    conv4 = Conv2D(256, (3, 3), padding = 'same')(conv3_pool)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4_pool = MaxPooling2D((2,2), strides=(2,2))(conv4)
    #16x16

    conv5 = Conv2D(512, (3, 3), padding = 'same')(conv4_pool)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up4 = UpSampling2D((2,2))(conv5)
    concatenate4 = concatenate([conv4, up4], axis=-1)
    #32x32

    dec4 = Conv2D(256, (3, 3), padding = 'same')(concatenate4)
    dec4 = BatchNormalization()(dec4)
    dec4 = Activation('relu')(dec4)

    up3 = UpSampling2D((2,2))(dec4)
    concatenate3 = concatenate([conv3, up3], axis=-1)
    #64x64

    dec3 = Conv2D(128, (3, 3), padding = 'same')(concatenate3)
    dec3 = BatchNormalization()(dec3)
    dec3 = Activation('relu')(dec3)

    up2 = UpSampling2D((2,2))(dec3)
    concatenate2 = concatenate([conv2, up2], axis=-1)
    #128x128

    dec2 = Conv2D(64, (3, 3), padding = 'same')(concatenate2)
    dec2 = BatchNormalization()(dec2)
    dec2 = Activation('relu')(dec2)

    up1 = UpSampling2D((2,2))(dec2)
    concatenate1 = concatenate([conv1, up1], axis=-1)
    #256x256

    dec1 = Conv2D(32, (3, 3), padding = 'same')(concatenate1)
    dec1 = BatchNormalization()(dec1)
    dec1 = Activation('relu')(dec1)

    outputs = Conv2D(1, 
                      kernel_size=1,
                      strides=1,
                      activation='sigmoid',
                      padding='same')(dec1)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    
    return model

if __name__ == '__main__':
    model = Unet()
    model.summary()
