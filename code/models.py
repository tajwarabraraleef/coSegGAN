'''
Code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''

from keras.models import *
from keras.layers import *
from keras.optimizers import *

def UNet(input_size):
    BF = 16

    inputs = Input(input_size)
    conv1 = Conv2D(BF, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    batch_norm1 = BatchNormalization()(conv1)

    conv1 = Conv2D(BF, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm1)
    batch_norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)

    conv2 = Conv2D(BF * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    batch_norm2 = BatchNormalization()(conv2)

    conv2 = Conv2D(BF * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm2)
    batch_norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)

    conv3 = Conv2D(BF * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    batch_norm3 = BatchNormalization()(conv3)

    conv3 = Conv2D(BF * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm3)
    batch_norm3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.2)(batch_norm3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(BF * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    batch_norm4 = BatchNormalization()(conv4)

    conv4 = Conv2D(BF * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm4)
    batch_norm4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.2)(batch_norm4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(BF * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    batch_norm5 = BatchNormalization()(conv5)
    drop5_ = Dropout(0.2)(batch_norm5)

    conv5 = Conv2D(BF * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop5_)
    batch_norm5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.2)(batch_norm5)

    up6 = Conv2D(BF * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(BF * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    batch_norm6 = BatchNormalization()(conv6)

    conv6 = Conv2D(BF * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm6)
    batch_norm6 = BatchNormalization()(conv6)

    up7 = Conv2D(BF * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch_norm6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(BF * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    batch_norm7 = BatchNormalization()(conv7)
    conv7 = Conv2D(BF * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm7)
    batch_norm7 = BatchNormalization()(conv7)

    up8 = Conv2D(BF * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch_norm7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(BF * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    batch_norm8 = BatchNormalization()(conv8)

    conv8 = Conv2D(BF * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm8)
    batch_norm8 = BatchNormalization()(conv8)

    up9 = Conv2D(BF, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch_norm8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(BF, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    batch_norm9 = BatchNormalization()(conv9)

    conv9 = Conv2D(BF, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm9)
    batch_norm9 = BatchNormalization()(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batch_norm9)  # why
    batch_norm9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(batch_norm9)

    model = Model(inputs=[inputs], outputs = [conv10])

    return model
