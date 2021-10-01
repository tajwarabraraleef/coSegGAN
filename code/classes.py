'''
Code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''

from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from imgaug import augmenters as iaa  # Install imgaug library (for data augmentation) from https://github.com/aleju/imgaug#installation
from keras.optimizers import *
import os
from tensorflow import keras
import data_utils
import pandas
import cv2

class DataGeneratorTraining():

    def __init__(self, hist_eq, Xdata_path_A, Xdata_path_B, Ydata_path, img_shape, shuffle=True, augment_flag=False):

        self.hist_eq = hist_eq
        self.shuffle = shuffle
        self.augment_flag = augment_flag
        self.img_shape = img_shape

        self.Xdata_path_A = Xdata_path_A
        self.Xdata_path_B = Xdata_path_B
        self.Ydata_path = Ydata_path

        # Augmentation types
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Flip Y-axis
            iaa.TranslateX(px=(-20, 20)),  # Translate along X axis by 20-20 pixels
            iaa.TranslateY(px=(-20, 20)),  # Trasnlate Y
            iaa.Rotate((-10, 10)),  # Rotate
            iaa.ScaleX((0.90, 1.15)),  # Along width 50%-150% of size
            iaa.ScaleY((0.90, 1.15))  # Along height
            ], random_order=True) #


    def load_batch(self, batch_size=1):

        # Finding number of batch based on the domain with the least amount of data
        self.n_batches = int(min(len(self.Xdata_path_A), len(self.Xdata_path_B))/ batch_size)
        total_samples = self.n_batches * batch_size

        self.indexes_A = np.arange(len(self.Xdata_path_A))
        self.indexes_B = np.arange(len(self.Xdata_path_B))

        if self.shuffle == True:
            np.random.shuffle(self.indexes_A)
            np.random.shuffle(self.indexes_B)

        X_A = [self.Xdata_path_A[i] for i in self.indexes_A[:total_samples]]
        X_B = [self.Xdata_path_B[i] for i in self.indexes_B[:total_samples]]
        y = [self.Ydata_path[i] for i in self.indexes_A[:total_samples]]

        # Enumerating batches of A and B
        for i in range(self.n_batches - 1):

            X_batch_A = X_A[i * batch_size:(i + 1) * batch_size]
            X_batch_B = X_B[i * batch_size:(i + 1) * batch_size]
            y_batch = y[i * batch_size:(i + 1) * batch_size]

            X_values_augmented_A, X_values_augmented_B, Y_values_augmented = ([] for _ in range(3))

            for Xa, Xb, Ya in zip(X_batch_A,X_batch_B, y_batch):

                x_temp_A = cv2.imread(Xa, 1)
                x_temp_A = cv2.cvtColor(x_temp_A, cv2.COLOR_BGR2RGB)
                if self.hist_eq == 'A':
                    #CLAHE hist eq for UCL
                    hsv = cv2.cvtColor(x_temp_A, cv2.COLOR_RGB2HSV)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
                    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
                    x_temp_A = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                x_temp_A = cv2.resize(x_temp_A, dsize=(self.img_shape[0:2]), interpolation=cv2.INTER_LINEAR)

                x_temp_B = cv2.imread(Xb, 1)
                x_temp_B = cv2.cvtColor(x_temp_B, cv2.COLOR_BGR2RGB)
                if self.hist_eq == 'B':
                    #CLAHE hist eq for UCL
                    hsv = cv2.cvtColor(x_temp_B, cv2.COLOR_RGB2HSV)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
                    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
                    x_temp_B = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                x_temp_B = cv2.resize(x_temp_B, dsize=(self.img_shape[0:2]), interpolation=cv2.INTER_LINEAR)

                y_temp = cv2.imread(Ya, 0)
                y_temp = np.expand_dims(cv2.resize(y_temp, dsize=(self.img_shape[0:2]), interpolation=cv2.INTER_LINEAR), axis=-1)
                y_temp = y_temp/255

                if self.augment_flag == True:
                    # Performs simultaneous augmentation on all inputs and segmentations
                    # The aumentations mentioned in seq is performed randomly in the order defined
                    seqA = self.seq.to_deterministic() # This makes the seq deterministic for this loop
                    x_temp_A = seqA.augment_image(x_temp_A)
                    y_temp = seqA.augment_image(y_temp)

                    seqB = self.seq.to_deterministic()  #This makes the seq deterministic for this loop
                    x_temp_B = seqB.augment_image(x_temp_B)

                X_values_augmented_A.append(x_temp_A)
                X_values_augmented_B.append(x_temp_B)
                Y_values_augmented.append(y_temp)

            X_augmented_A = np.asarray(X_values_augmented_A)/127.5 - 1  #255/127.5 = 0-2 => -1:1
            X_augmented_B = np.asarray(X_values_augmented_B)/127.5 - 1
            Y_augmented = np.asarray(Y_values_augmented) >= 0.5

            indx = np.arange(len(X_augmented_A))
            np.random.shuffle(indx)
            X_augmented_A = X_augmented_A[indx]
            X_augmented_B = X_augmented_B[indx]
            Y_augmented = Y_augmented[indx]

            yield X_augmented_A, X_augmented_B, Y_augmented


class DataGeneratorValidation():

    def __init__(self, hist_eq, Xdata_path, Ydata_path, img_shape, shuffle=True, augment_flag=False):

        'Initialization'
        self.shuffle = shuffle
        self.augment_flag = augment_flag
        self.img_shape = img_shape
        self.Xdata_path = Xdata_path
        self.Ydata_path = Ydata_path
        self.hist_eq = hist_eq

        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Flip Y-axis
            iaa.TranslateX(px=(-20, 20)),  # Translate along X axis by 20-20 pixels
            iaa.TranslateY(px=(-20, 20)),  # Trasnlate Y
            iaa.Rotate((-10, 10)),  # Rotate
            iaa.ScaleX((0.90, 1.15)),  # Along width 50%-150% of size
            iaa.ScaleY((0.90, 1.15))  # Along height
            ], random_order=True) #

    def load_all(self, batch_size=1):

        # Finding number of batches based on the domain with the least amount of data
        self.n_batches = int(len(self.Xdata_path) / batch_size)
        total_samples = self.n_batches * batch_size
        self.indexes = np.arange(len(self.Xdata_path))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        X = [self.Xdata_path[i] for i in self.indexes[:total_samples]]
        y = [self.Ydata_path[i] for i in self.indexes[:total_samples]]

        X_values_augmented ,Y_values_augmented = ([] for _ in range(2))

        # Enumerating batches of A and B
        for i in range(len(X)):
            x_temp = cv2.imread(X[i], 1)
            x_temp = cv2.cvtColor(x_temp, cv2.COLOR_BGR2RGB)
            if self.hist_eq:
                # CLAHE hist eq for UCL
                hsv = cv2.cvtColor(x_temp, cv2.COLOR_RGB2HSV)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
                hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
                x_temp = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            x_temp = cv2.resize(x_temp, dsize=(self.img_shape[0:2]), interpolation=cv2.INTER_LINEAR)

            y_temp = cv2.imread(y[i], 0)
            y_temp = np.expand_dims(cv2.resize(y_temp, dsize=(self.img_shape[0:2]), interpolation=cv2.INTER_LINEAR), axis=-1)
            y_temp = y_temp/255

            if self.augment_flag == True:
                seq1 = self.seq.to_deterministic()  #This makes the seq deterministic for this loop
                x_temp = seq1.augment_image(x_temp)
                y_temp = seq1.augment_image(y_temp)

            X_values_augmented.append(x_temp)
            Y_values_augmented.append(y_temp)

        X_augmented = np.asarray(X_values_augmented)/127.5 - 1
        Y_augmented = np.asarray(Y_values_augmented) >= 0.5

        indx = np.arange(len(X_augmented))
        np.random.shuffle(indx)
        X_augmented = X_augmented[indx]
        Y_augmented = Y_augmented[indx]

        return X_augmented, Y_augmented



