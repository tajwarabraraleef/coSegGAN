'''
Training code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''

from __future__ import print_function, division
from keras.layers import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.utils import generic_utils
import models
import metrics
import classes
from data_path_loader import data_path_loader
from scipy.stats import iqr
import data_utils
from imgaug import augmenters as iaa
import tensorflow as tf

class coSegGAN():
    def __init__(self):

        self.data_A_name = 'Endovis' #Can be any date with labels
        self.data_B_name = 'Surgery' #Data without labels
        augment_flag = True
        loss_shape = metrics.binary_focal_loss() # Segmentation model loss

        # Create directory
        self.model_name = '../models/Model with Domain A_' + self.data_A_name + ' & B_' + self.data_B_name
        os.makedirs('%s/Generator eval/' % (self.model_name), exist_ok=True)

        # Perform histEQ if UCL data is used (because data has inconsistent lighting)
        hist_eq = '_'
        hist_eq_A = False
        hist_eq_B = False

        if self.data_A_name == 'UCL':
            hist_eq = 'A'
            hist_eq_A = True
        elif self.data_B_name == 'UCL':
            hist_eq = 'B'
            hist_eq_B = True

        self.operating_point = 0.5# Threshold on output for converting to binary image

        # Input shape of RGB frames
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Load paths of all frames in Domain A
        [self.Xa, self.Y, self.Xa_val, self.Ya_val] = data_path_loader(self.data_A_name)
        # Load paths of all frames in Domain B
        [self.Xb, _, self.Xb_val, self.Yb_val] = data_path_loader(self.data_B_name) # Theres no label for Domain B training. But for validation, labels are needed

        print('\nData A: ' + self.data_A_name)
        print('Xa: ' + str(len(self.Xa)))
        print('Y: ' + str(len(self.Y)))
        print('Xa_val: ' + str(len(self.Xa_val)))
        print('Ya_val: ' + str(len(self.Ya_val)))

        print('\nData B: ' + self.data_B_name)
        print('Xb: ' + str(len(self.Xb)))
        print('Xb_val: ' + str(len(self.Xb_val)))
        print('Yb_val: ' + str(len(self.Yb_val)))

        # Data loader for Training data
        self.data_loader_train = classes.DataGeneratorTraining(hist_eq, self.Xa, self.Xb, self.Y, self.img_shape, shuffle=True, augment_flag=augment_flag)

        # Data loader for Validation data
        self.data_loader_val_B = classes.DataGeneratorValidation(hist_eq_B, self.Xb_val, self.Yb_val, self.img_shape, shuffle=True, augment_flag=False)
        self.data_loader_val_A = classes.DataGeneratorValidation(hist_eq_A, self.Xa_val, self.Ya_val, self.img_shape, shuffle=True, augment_flag=False)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_1 = 1.0     # GAN total loss
        self.lambda_2 = 10.0    # Cycle-consistency loss
        self.lambda_3 = 1.0     # Shape loss
        self.lambda_4 = 5.0     # Structure loss
        self.lambda_5 = 0.1 * self.lambda_2    # Identity loss
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.S = models.UNet(self.img_shape)
        self.S.summary()
        print(self.model_name)

        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.S.compile(loss=loss_shape,
                       optimizer=Adam(lr=1e-3),
                       metrics=[loss_shape, metrics.dice_coef_metric])

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B, encoded_img_A = self.g_AB(img_A)
        fake_A, encoded_img_B = self.g_BA(img_B)

        # Translate images back to original domain
        reconstr_A, encoded_fake_B = self.g_BA(fake_B)
        reconstr_B, encoded_fake_A = self.g_AB(fake_A)

        encoded_img_A = Lambda(lambda xx: tf.expand_dims(xx, axis=-1))(encoded_img_A)
        encoded_fake_B = Lambda(lambda xx: tf.expand_dims(xx, axis=-1))(encoded_fake_B)
        encoded_img_B = Lambda(lambda xx: tf.expand_dims(xx, axis=-1))(encoded_img_B)
        encoded_fake_A = Lambda(lambda xx: tf.expand_dims(xx, axis=-1))(encoded_fake_A)

        all_encoded = Concatenate(axis=-1)([encoded_img_A, encoded_fake_B, encoded_img_B, encoded_fake_A])

        # Identity mapping of images
        img_A_id, _ = self.g_BA(img_A)
        img_B_id, _ = self.g_AB(img_B)

        # For the coSegGAN model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False
        self.S.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # (S uses 0-1 range dat so converting data accordingly)
        fake_B_for_unet = Lambda(lambda xx: xx * 0.5 + 0.5)(fake_B)
        S_A2B = self.S(fake_B_for_unet) #Labels will be same as Domain A

        # coSegGAN
        self.coSegGAN = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id, S_A2B, all_encoded])
        self.coSegGAN.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae', loss_shape, metrics.loss_structure],
                              loss_weights=[self.lambda_1, self.lambda_1,
                                            self.lambda_2, self.lambda_2,
                                            self.lambda_5, self.lambda_5, self.lambda_3, self.lambda_4],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(inputs=d0, outputs=[output_img, d4])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=10):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        dummy_value = np.zeros((batch_size, 16, 16, 256, 4))  # Depends on the size of the encoded vector (d4 from generator), dummy values used to bypass the input of keras train function.
        #in reality, the structure loss doesnt need ground truth but rather compared between encoded vectors of Ga and Gb

        self.n_batches = int(
            min(len(self.Xa), len(self.Xb)) / batch_size)  # this is based on which dataset has min data, the network trains for that many samples only

        # Make a log file
        record_df = pd.DataFrame(
            columns=['epoch', 'Unet_loss', 'd_Loss', 'd_accuracy', 'g_loss', 'g_adv', 'g_recon', 'g_id', 'g_unet', 'g_encoded', 'elapsed_time'])

        record_df_train_A = pd.DataFrame(
            columns=['epoch', 'IOU Mean', 'IOU Median', 'IOU IQR', 'Dice Mean', 'Dice Median', 'Dice IQR', 'Sen Mean', 'Sen Median', 'Sen IQR', 'Spe Mean',
                     'Spe Median', 'Spe IQR', 'elapsed_time'])

        record_df_val_A = pd.DataFrame(
            columns=['epoch', 'IOU Mean', 'IOU Median', 'IOU IQR', 'Dice Mean', 'Dice Median', 'Dice IQR', 'Sen Mean', 'Sen Median', 'Sen IQR', 'Spe Mean',
                     'Spe Median', 'Spe IQR', 'elapsed_time'])
        record_df_val_B = pd.DataFrame(
            columns=['epoch', 'IOU Mean', 'IOU Median', 'IOU IQR', 'Dice Mean', 'Dice Median', 'Dice IQR', 'Sen Mean', 'Sen Median', 'Sen IQR', 'Spe Mean',
                     'Spe Median', 'Spe IQR', 'elapsed_time'])

        # Setting to a negative number
        iou_check_mean_A = -10000
        dice_check_mean_A = -10000
        iou_check_median_A = -10000
        dice_check_median_A = -10000

        iou_check_mean_B = -10000
        dice_check_mean_B = -10000
        iou_check_median_B = -10000
        dice_check_median_B = -10000

        iou_mean_train_A, dice_mean_train_A, sen_mean_train_A, spe_mean_train_A = ([] for _ in range(4))
        iou_median_train_A, dice_median_train_A, sen_median_train_A, spe_median_train_A = ([] for _ in range(4))
        iou_iqr_train_A, dice_iqr_train_A, sen_iqr_train_A, spe_iqr_train_A = ([] for _ in range(4))

        iou_mean_val_A, dice_mean_val_A, sen_mean_val_A, spe_mean_val_A = ([] for _ in range(4))
        iou_median_val_A, dice_median_val_A, sen_median_val_A, spe_median_val_A = ([] for _ in range(4))
        iou_iqr_val_A, dice_iqr_val_A, sen_iqr_val_A, spe_iqr_val_A = ([] for _ in range(4))

        iou_mean_val_B, dice_mean_val_B, sen_mean_val_B, spe_mean_val_B = ([] for _ in range(4))
        iou_median_val_B, dice_median_val_B, sen_median_val_B, spe_median_val_B = ([] for _ in range(4))
        iou_iqr_val_B, dice_iqr_val_B, sen_iqr_val_B, spe_iqr_val_B = ([] for _ in range(4))

        epochs_count = []

        for epoch in range(epochs):
            progbar = generic_utils.Progbar(self.n_batches * batch_size)

            specificity, sensitivity, dice, iou = ([] for _ in range(4))

            for batch_i, (imgs_A, imgs_B, imgs_y) in enumerate(self.data_loader_train.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators and Segmentation model
                # ----------------------

                # Translate images to opposite domain
                fake_B, _ = self.g_AB.predict(imgs_A)
                fake_A, _ = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake) and Segmentation model
                self.d_A.trainable = True
                self.d_B.trainable = True
                self.S.trainable = True

                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                ## S training part
                S_input = np.concatenate((imgs_A, fake_B), axis=0)
                S_out = np.concatenate((imgs_y, imgs_y), axis=0)

                # Shuffle
                self.indexes = np.arange(len(S_input))
                np.random.shuffle(self.indexes)
                S_input = (S_input[self.indexes]) * 0.5 + 0.5  # 0-1
                S_out = S_out[self.indexes]

                # Training with batches of images for S instead of all images together
                n_batches_S = int(len(S_input) / batch_size)
                S_loss = []
                for i in range(n_batches_S - 1):
                    S_input_batch = S_input[i * batch_size:(i + 1) * batch_size]
                    S_out_batch = S_out[i * batch_size:(i + 1) * batch_size]
                    S_loss.append(self.S.train_on_batch(S_input_batch, S_out_batch))

                S_loss = np.mean(S_loss, axis=0).tolist()

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.coSegGAN.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B, imgs_y, dummy_value])


                # Prediction results
                pred_y = self.S.predict(S_input)

                specificity_batch, sensitivity_batch, dice_batch, iou_batch = data_utils.use_operating_points(self.operating_point, S_out, pred_y)

                elapsed_time = datetime.datetime.now() - start_time
                progbar.add(batch_size, values=[("Epoch", np.uint8(epoch)), ('D loss', d_loss[0]), ('G loss', g_loss[0]), ('Encoded loss', g_loss[-1]),
                                                ('S loss', S_loss[0]), ("Mean IOU", np.mean(iou_batch) * 100), ("Mean DICE", np.mean(dice_batch) * 100),
                                                ("Mean Sensitivity", np.mean(sensitivity_batch) * 100), ("Mean Specificity", np.mean(specificity_batch) * 100)])

                iou = iou + iou_batch
                dice = dice + dice_batch
                sensitivity = sensitivity + sensitivity_batch
                specificity = specificity + specificity_batch

            iou_mean_train_A.append(np.mean(iou))
            dice_mean_train_A.append(np.mean(dice))
            sen_mean_train_A.append(np.mean(sensitivity))
            spe_mean_train_A.append(np.mean(specificity))

            iou_median_train_A.append(np.median(iou))
            dice_median_train_A.append(np.median(dice))
            sen_median_train_A.append(np.median(sensitivity))
            spe_median_train_A.append(np.median(specificity))

            iou_iqr_train_A.append(iqr(iou))
            dice_iqr_train_A.append(iqr(dice))
            sen_iqr_train_A.append(iqr(sensitivity))
            spe_iqr_train_A.append(iqr(specificity))
            epochs_count.append(epoch)

            # Plot the progress
            print("\n\n[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, "
                  "id: %05f] [S loss: %f] time: %s \n" % (
                epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), S_loss[0],
                elapsed_time))
            print(self.model_name)
            # Log metrics at end of epoch
            new_row = {'epoch': epoch, 'Unet_loss': S_loss[0], 'd_Loss': d_loss[0], 'd_accuracy': 100 * d_loss[1], 'g_loss': g_loss[0],
                       'g_adv': np.mean(g_loss[1:3]), 'g_recon': np.mean(g_loss[3:5]), 'g_id': np.mean(g_loss[5:6]), 'g_unet': g_loss[-2],
                       'g_encoded': g_loss[-1], 'elapsed_time': elapsed_time}

            record_df = record_df.append(new_row, ignore_index=True)
            record_df.to_csv("%s/Model_loss.csv" % (self.model_name), index=0)

            print('Training A and Fake_B Combined: Mean-> [Median]-> (IQR) IOU: %0.1f-> [%0.1f]-> (%0.1f), Dice: %0.1f-> [%0.1f]-> ('
                  '%0.1f), '
                  'Sensi: %0.1f-> [%0.1f]-> (%0.1f), Speci: %0.1f-> [%0.1f]-> (%0.1f)' % (
                      iou_mean_train_A[epoch] * 100, iou_median_train_A[epoch] * 100, iou_iqr_train_A[epoch] * 100, dice_mean_train_A[epoch] * 100,
                      dice_median_train_A[epoch] * 100, dice_iqr_train_A[epoch] * 100, sen_mean_train_A[epoch] * 100, sen_median_train_A[epoch] * 100,
                      sen_iqr_train_A[epoch] * 100, spe_mean_train_A[epoch] * 100, spe_median_train_A[epoch] * 100, spe_iqr_train_A[epoch] * 100))

            # Log metrics at end of epoch
            new_row_train = {'epoch': epoch, 'IOU Mean': iou_mean_train_A[epoch], 'IOU Median': iou_median_train_A[epoch], 'IOU IQR': iou_iqr_train_A[epoch],
                             'Dice Mean': dice_mean_train_A[epoch], 'Dice Median': dice_median_train_A[epoch], 'Dice IQR': dice_iqr_train_A[epoch],
                             'Sen Mean': sen_mean_train_A[epoch], 'Sen Median': sen_median_train_A[epoch], 'Sen IQR': sen_iqr_train_A[epoch],
                             'Spe Mean': spe_mean_train_A[epoch], 'Spe Median': spe_median_train_A[epoch], 'Spe IQR': spe_iqr_train_A[epoch],
                             'elapsed_time': elapsed_time}

            record_df_train_A = record_df_train_A.append(new_row_train, ignore_index=True)
            record_df_train_A.to_csv("%s/Metrics_train_A_and_fakeB_combined.csv" % (self.model_name), index=0)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.Generator_image_plot(epoch, imgs_A[0:1], imgs_B[0:1], imgs_y[0:1], 'train_A_fakeB_combined')

            if epoch == 0:  # only need to load once

                # Check validation results
                (imgs_B_val, imgs_Yb_val) = (self.data_loader_val_B.load_all(len(self.Xb_val)))
                (imgs_A_val, imgs_Ya_val) = (self.data_loader_val_A.load_all(len(self.Xa_val)))

                print('Xa_val Max: %0.1f, Min: %0.1f' % (np.max(imgs_A_val), np.min(imgs_A_val)))
                print('Ya_val Max: %0.1f, Min: %0.1f' % (np.max(imgs_Ya_val), np.min(imgs_Ya_val)))

                print('Xb_val Max: %0.1f, Min: %0.1f' % (np.max(imgs_B_val), np.min(imgs_B_val)))
                print('Yb_val Max: %0.1f, Min: %0.1f' % (np.max(imgs_Yb_val), np.min(imgs_Yb_val)))

                imgs_A_val = imgs_A_val * 0.5 + 0.5
                imgs_B_val = imgs_B_val * 0.5 + 0.5

            # Shuffle
            self.indexes = np.arange(len(imgs_A_val))
            np.random.shuffle(self.indexes)
            imgs_A_val = (imgs_A_val[self.indexes])  # 0-1
            imgs_Ya_val = (imgs_Ya_val[self.indexes])

            self.indexes = np.arange(len(imgs_B_val))
            np.random.shuffle(self.indexes)
            imgs_B_val = (imgs_B_val[self.indexes])  # 0-1
            imgs_Yb_val = (imgs_Yb_val[self.indexes])

            pred_val_A = self.S.predict(imgs_A_val)
            pred_val_B = self.S.predict(imgs_B_val)

            if epoch % sample_interval == 0:
                self.segmentation_eval(epoch, imgs_A_val[:6], imgs_Ya_val[:6], 'Val_A')
                self.segmentation_eval(epoch, imgs_B_val[:6], imgs_Yb_val[:6], 'Val_B')

            specificity, sensitivity, dice, iou = data_utils.use_operating_points(self.operating_point, imgs_Ya_val, pred_val_A)

            iou_mean_val_A.append(np.mean(iou))
            dice_mean_val_A.append(np.mean(dice))
            sen_mean_val_A.append(np.mean(sensitivity))
            spe_mean_val_A.append(np.mean(specificity))

            iou_median_val_A.append(np.median(iou))
            dice_median_val_A.append(np.median(dice))
            sen_median_val_A.append(np.median(sensitivity))
            spe_median_val_A.append(np.median(specificity))

            iou_iqr_val_A.append(iqr(iou))
            dice_iqr_val_A.append(iqr(dice))
            sen_iqr_val_A.append(iqr(sensitivity))
            spe_iqr_val_A.append(iqr(specificity))

            specificity, sensitivity, dice, iou = data_utils.use_operating_points(self.operating_point, imgs_Yb_val, pred_val_B)

            iou_mean_val_B.append(np.mean(iou))
            dice_mean_val_B.append(np.mean(dice))
            sen_mean_val_B.append(np.mean(sensitivity))
            spe_mean_val_B.append(np.mean(specificity))

            iou_median_val_B.append(np.median(iou))
            dice_median_val_B.append(np.median(dice))
            sen_median_val_B.append(np.median(sensitivity))
            spe_median_val_B.append(np.median(specificity))

            iou_iqr_val_B.append(iqr(iou))
            dice_iqr_val_B.append(iqr(dice))
            sen_iqr_val_B.append(iqr(sensitivity))
            spe_iqr_val_B.append(iqr(specificity))

            print('Validation A: Mean-> [Median]-> (IQR) IOU: %0.1f-> [%0.1f]-> (%0.1f), Dice: %0.1f-> [%0.1f]-> (%0.1f), '
                  'Sensi: %0.1f-> [%0.1f]-> (%0.1f), Speci: %0.1f-> [%0.1f]-> (%0.1f)' % (
                      iou_mean_val_A[epoch] * 100, iou_median_val_A[epoch] * 100, iou_iqr_val_A[epoch] * 100, dice_mean_val_A[epoch] * 100,
                      dice_median_val_A[epoch] * 100, dice_iqr_val_A[epoch] * 100, sen_mean_val_A[epoch] * 100, sen_median_val_A[epoch] * 100,
                      sen_iqr_val_A[epoch] * 100, spe_mean_val_A[epoch] * 100, spe_median_val_A[epoch] * 100, spe_iqr_val_A[epoch] * 100))

            print('Validation B (unseen): Mean-> [Median]-> (IQR) IOU: %0.1f-> [%0.1f]-> (%0.1f), Dice: %0.1f-> [%0.1f]-> (%0.1f), '
                  'Sensi: %0.1f-> [%0.1f]-> (%0.1f), Speci: %0.1f-> [%0.1f]-> (%0.1f)' % (
                      iou_mean_val_B[epoch] * 100, iou_median_val_B[epoch] * 100, iou_iqr_val_B[epoch] * 100, dice_mean_val_B[epoch] * 100,
                      dice_median_val_B[epoch] * 100, dice_iqr_val_B[epoch] * 100, sen_mean_val_B[epoch] * 100, sen_median_val_B[epoch] * 100,
                      sen_iqr_val_B[epoch] * 100, spe_mean_val_B[epoch] * 100, spe_median_val_B[epoch] * 100, spe_iqr_val_B[epoch] * 100))

            print('\nMAX on Validation A: Mean (Median)-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)\n'
                  'MAX on Validation B (unseen): Mean (Median)-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)\n' % (
                      np.max(iou_mean_val_A) * 100, np.max(iou_median_val_A) * 100, np.max(dice_mean_val_A) * 100, np.max(dice_median_val_A) * 100,
                      np.max(iou_mean_val_B) * 100, np.max(iou_median_val_B) * 100, np.max(dice_mean_val_B) * 100, np.max(dice_median_val_B) * 100))

            # Log metrics at end of epoch
            new_row_val_A = {'epoch': epoch, 'IOU Mean': iou_mean_val_A[epoch], 'IOU Median': iou_median_val_A[epoch], 'IOU IQR': iou_iqr_val_A[epoch],
                             'Dice Mean': dice_mean_val_A[epoch], 'Dice Median': dice_median_val_A[epoch], 'Dice IQR': dice_iqr_val_A[epoch],
                             'Sen Mean': sen_mean_val_A[epoch], 'Sen Median': sen_median_val_A[epoch], 'Sen IQR': sen_iqr_val_A[epoch],
                             'Spe Mean': spe_mean_val_A[epoch], 'Spe Median': spe_median_val_A[epoch], 'Spe IQR': spe_iqr_val_A[epoch],
                             'elapsed_time': elapsed_time}

            record_df_val_A = record_df_val_A.append(new_row_val_A, ignore_index=True)
            record_df_val_A.to_csv("%s/Metrics_val_A.csv" % (self.model_name), index=0)

            # Log metrics at end of epoch
            new_row_val_B = {'epoch': epoch, 'IOU Mean': iou_mean_val_B[epoch], 'IOU Median': iou_median_val_B[epoch], 'IOU IQR': iou_iqr_val_B[epoch],
                             'Dice Mean': dice_mean_val_B[epoch], 'Dice Median': dice_median_val_B[epoch], 'Dice IQR': dice_iqr_val_B[epoch],
                             'Sen Mean': sen_mean_val_B[epoch], 'Sen Median': sen_median_val_B[epoch], 'Sen IQR': sen_iqr_val_B[epoch],
                             'Spe Mean': spe_mean_val_B[epoch], 'Spe Median': spe_median_val_B[epoch], 'Spe IQR': spe_iqr_val_B[epoch],
                             'elapsed_time': elapsed_time}

            record_df_val_B = record_df_val_B.append(new_row_val_B, ignore_index=True)
            record_df_val_B.to_csv("%s/Metrics_val_B.csv" % (self.model_name), index=0)


            self.plot_training_history(epochs_count, iou_mean_train_A, iou_median_train_A, iou_iqr_train_A, dice_mean_train_A, dice_median_train_A,
                                       dice_iqr_train_A, sen_mean_train_A, sen_median_train_A, sen_iqr_train_A, spe_mean_train_A, spe_median_train_A,
                                       spe_iqr_train_A, iou_mean_val_A, iou_median_val_A, iou_iqr_val_A, dice_mean_val_A, dice_median_val_A, dice_iqr_val_A,
                                       sen_mean_val_A, sen_median_val_A, sen_iqr_val_A, spe_mean_val_A, spe_median_val_A, spe_iqr_val_A, iou_mean_val_B,
                                       iou_median_val_B, iou_iqr_val_B, dice_mean_val_B, dice_median_val_B, dice_iqr_val_B, sen_mean_val_B, sen_median_val_B,
                                       sen_iqr_val_B, spe_mean_val_B, spe_median_val_B, spe_iqr_val_B)

            # Saving weights based on many criterions, modify as needed
            if iou_mean_val_A[epoch] > iou_check_mean_A:
                iou_check_mean_A = iou_mean_val_A[epoch]
                print("Saving IOU Mean A model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_mean_iou_A.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_mean_iou_A.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_mean_iou_A.h5"))

            if iou_median_val_A[epoch] > iou_check_median_A:
                iou_check_median_A = iou_median_val_A[epoch]
                print("Saving IOU Median A model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_median_iou_A.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_median_iou_A.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_median_iou_A.h5"))

            if dice_mean_val_A[epoch] > dice_check_mean_A:
                dice_check_mean_A = dice_mean_val_A[epoch]
                print("Saving DICE Mean A model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_mean_dice_A.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_mean_dice_A.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_mean_dice_A.h5"))

            if dice_median_val_A[epoch] > dice_check_median_A:
                dice_check_median_A = dice_median_val_A[epoch]
                print("Saving DICE Median A model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_median_dice_A.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_median_dice_A.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_median_dice_A.h5"))

            if iou_mean_val_B[epoch] > iou_check_mean_B:
                iou_check_mean_B = iou_mean_val_B[epoch]
                print("Saving IOU Mean B model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_mean_iou_B.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_mean_iou_B.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_mean_iou_B.h5"))

            if iou_median_val_B[epoch] > iou_check_median_B:
                iou_check_median_B = iou_median_val_B[epoch]
                print("Saving IOU Median B model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_median_iou_B.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_median_iou_B.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_median_iou_B.h5"))

            if dice_mean_val_B[epoch] > dice_check_mean_B:
                dice_check_mean_B = dice_mean_val_B[epoch]
                print("Saving DICE Mean B model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_mean_dice_B.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_mean_dice_B.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_mean_dice_B.h5"))

            if dice_median_val_B[epoch] > dice_check_median_B:
                dice_check_median_B = dice_median_val_B[epoch]
                print("Saving DICE Median B model at {} epoch.".format(epoch))
                self.S.save(filepath='%s/%s' % (self.model_name, "S_best_median_dice_B.h5"))
                self.g_AB.save(filepath='%s/%s' % (self.model_name, "A2B_best_median_dice_B.h5"))
                self.g_BA.save(filepath='%s/%s' % (self.model_name, "B2A_best_median_dice_B.h5"))

    def segmentation_eval(self, epoch, imgs_val, Y, type):

        if type == 'Val_A':
            X_org = imgs_val
            X_trans, _ = self.g_AB.predict((X_org - 0.5) / 0.5)
            X_iden, _ = self.g_BA.predict((X_org - 0.5) / 0.5)
        else:
            X_org = imgs_val
            X_trans, _ = self.g_BA.predict((X_org - 0.5) / 0.5)
            X_iden, _ = self.g_AB.predict((X_org - 0.5) / 0.5)

        Y_pred = self.S.predict(X_org)
        Y_pred_trans = self.S.predict(X_trans * 0.5 + 0.5)
        Y_pred_iden = self.S.predict(X_iden * 0.5 + 0.5)

        Y_pred = np.asarray(Y_pred)
        Y_pred_trans = np.asarray(Y_pred_trans)
        Y_pred_iden = np.asarray(Y_pred_iden)

        save_path = os.path.join(self.model_name, 'Segmentation eval')
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(11, 7), dpi=160)

        for i in range(len(X_org)):
            img = X_org[i]
            fake = X_trans[i] * 0.5 + 0.5
            img_iden = X_iden[i] * 0.5 + 0.5

            pred = Y_pred[i, :, :, 0]
            pred_fake = Y_pred_trans[i, :, :, 0]
            pred_iden = Y_pred_iden[i, :, :, 0]
            y = Y[i, :, :, 0]
            pred_avg = (pred + pred_fake + pred_iden) / 3

            ax = plt.subplot(6, 11, 11 * i + 1)
            plt.imshow(img)
            ax.set_title('Org ' + str(i), size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 2)
            plt.imshow(pred, cmap='gray', vmin=0, vmax=1.0)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])
            _, _, dice, iou = data_utils.use_operating_points(self.operating_point, Y[i:i + 1], Y_pred[i:i + 1])
            ax.set_title('IO:%0.1f, DI:%0.1f' % (iou[0] * 100, dice[0] * 100), size=6)

            ax = plt.subplot(6, 11, 11 * i + 3)
            plt.imshow(abs(y - pred), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title('Diff', size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 4)
            plt.imshow(fake)
            ax.set_title('Trans ' + str(i), size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 5)
            plt.imshow(pred_fake, cmap='gray', vmin=0, vmax=1.0)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])
            _, _, dice, iou = data_utils.use_operating_points(self.operating_point, Y[i:i + 1], Y_pred_trans[i:i + 1])
            ax.set_title('IO:%0.1f, DI:%0.1f' % (iou[0] * 100, dice[0] * 100), size=6)

            ax = plt.subplot(6, 11, 11 * i + 6)
            plt.imshow(abs(y - pred_fake), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title('Diff', size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 7)
            plt.imshow(img_iden)
            ax.set_title('Iden ' + str(i), size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 8)
            plt.imshow(pred_iden, cmap='gray', vmin=0, vmax=1.0)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])
            _, _, dice, iou = data_utils.use_operating_points(self.operating_point, Y[i:i + 1], Y_pred_iden[i:i + 1])
            ax.set_title('IO:%0.1f, DI:%0.1f' % (iou[0] * 100, dice[0] * 100), size=6)

            ax = plt.subplot(6, 11, 11 * i + 9)
            plt.imshow(abs(y - pred_iden), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title('Diff', size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

            ax = plt.subplot(6, 11, 11 * i + 10)
            plt.imshow(pred_avg, cmap='gray', vmin=0, vmax=1.0)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])
            _, _, dice, iou = data_utils.use_operating_points(self.operating_point, Y[i:i + 1], np.expand_dims(np.expand_dims(pred_avg, axis=0), axis=-1))
            ax.set_title('IO:%0.1f, DI:%0.1f' % (iou[0] * 100, dice[0] * 100), size=6)

            ax = plt.subplot(6, 11, 11 * i + 11)
            plt.imshow(abs(y - pred_avg), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title('Diff Avg ' + str(i), size=6)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(True)
            ax.get_xaxis().set_ticks([])
            ax.get_xaxis().set_ticklabels([])

        _, _, dice_org, iou_org = data_utils.use_operating_points(self.operating_point, Y, Y_pred)
        _, _, dice_trans, iou_trans = data_utils.use_operating_points(self.operating_point, Y, Y_pred_trans)
        _, _, dice_iden, iou_iden = data_utils.use_operating_points(self.operating_point, Y, Y_pred_iden)
        _, _, dice_avg, iou_avg = data_utils.use_operating_points(self.operating_point, Y, (Y_pred + Y_pred_trans + Y_pred_iden) / 3)

        plt.suptitle('Data type, %s: Mean (Median)\n'
                     'Org-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)'
                     ', Trans-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)'
                     '\nIden-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)'
                     ', Avg-> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)' % (
                         type, np.mean(iou_org) * 100, np.median(iou_org) * 100, np.mean(dice_org) * 100, np.median(dice_org) * 100, np.mean(iou_trans) * 100,
                         np.median(iou_trans) * 100, np.mean(dice_trans) * 100, np.median(dice_trans) * 100, np.mean(iou_iden) * 100, np.median(iou_iden) * 100,
                         np.mean(dice_iden) * 100, np.median(dice_iden) * 100, np.mean(iou_avg) * 100, np.median(iou_avg) * 100, np.mean(dice_avg) * 100,
                         np.median(dice_avg) * 100))

        plt.savefig(save_path + ("/%d_%s.png") % (epoch, type))
        plt.clf()
        plt.close()

    def Generator_image_plot(self, epoch, imgs_A, imgs_B, imgs_y, type):

        r, c = 4, 4

        # Translate images to the other domain
        fake_B, _ = self.g_AB.predict(imgs_A)
        fake_A, _ = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A, _ = self.g_BA.predict(fake_B)
        reconstr_B, _ = self.g_AB.predict(fake_A)

        iden_A, _ = self.g_BA.predict(imgs_A)
        iden_B, _ = self.g_AB.predict(imgs_B)

        S_imgs_A = self.S.predict(imgs_A * 0.5 + 0.5)
        S_imgs_B = self.S.predict(imgs_B * 0.5 + 0.5)

        S_fake_B = self.S.predict(fake_B * 0.5 + 0.5)
        S_fake_A = self.S.predict(fake_A * 0.5 + 0.5)

        S_reconstr_A = self.S.predict(reconstr_A * 0.5 + 0.5)
        S_reconstr_B = self.S.predict(reconstr_B * 0.5 + 0.5)

        S_iden_A = self.S.predict(iden_A * 0.5 + 0.5)
        S_iden_B = self.S.predict(iden_B * 0.5 + 0.5)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, iden_A, imgs_B, fake_A, reconstr_B, iden_B])
        gen_segs = np.concatenate([S_imgs_A, S_fake_B, S_reconstr_A, S_iden_A, S_imgs_B, S_fake_A, S_reconstr_B, S_iden_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles_1 = ['A', 'fakeB', 'Reconstructed', 'Identity']
        titles_2 = ['B', 'fakeA', 'Reconstructed', 'Identity']

        _, _, dice_A, iou_A = data_utils.use_operating_points(self.operating_point, imgs_y, S_imgs_A)
        _, _, dice_fakeB, iou_fakeB = data_utils.use_operating_points(self.operating_point, imgs_y, S_fake_B)
        _, _, dice_recons, iou_recons = data_utils.use_operating_points(self.operating_point, imgs_y, S_reconstr_A)
        _, _, dice_id, iou_id = data_utils.use_operating_points(self.operating_point, imgs_y, S_iden_A)

        metrics = ["IOU: %0.1f, Dice: %0.1f" % (iou_A[0] * 100, dice_A[0] * 100), "IOU: %0.1f, Dice: %0.1f" % (iou_fakeB[0] * 100, dice_fakeB[0] * 100),
                   "IOU: %0.1f, Dice: %0.1f" % (iou_recons[0] * 100, dice_recons[0] * 100), "IOU: %0.1f, Dice: %0.1f" % (iou_id[0] * 100, dice_id[0] * 100)]

        fig, axs = plt.subplots(r, c, dpi=160)
        cnt = 0
        cnt_seg = 0
        for i in range(r):
            for j in range(c):
                if (i == 0 or i == 2):
                    axs[i, j].imshow(gen_imgs[cnt])
                    cnt += 1
                    if i == 0:
                        axs[i, j].set_title(titles_1[j], size=5)
                    if i == 2:
                        axs[i, j].set_title(titles_2[j], size=5)
                if (i == 1 or i == 3):
                    axs[i, j].imshow(gen_segs[cnt_seg, :, :, 0], cmap='gray', vmin=0, vmax=1.0)
                    cnt_seg += 1
                    if i == 1:
                        axs[i, j].set_title(metrics[j], size=5)

                axs[i, j].axis('off')

        fig.savefig("%s/Generator eval/%d_%s.png" % (self.model_name, epoch, type))
        plt.close()

    def plot_training_history(self, epochs_count, iou_mean_train, iou_median_train, iou_iqr_train, dice_mean_train, dice_median_train, dice_iqr_train,
                              sen_mean_train, sen_median_train, sen_iqr_train, spe_mean_train, spe_median_train, spe_iqr_train, iou_mean_val_A,
                              iou_median_val_A, iou_iqr_val_A, dice_mean_val_A, dice_median_val_A, dice_iqr_val_A, sen_mean_val_A, sen_median_val_A,
                              sen_iqr_val_A, spe_mean_val_A, spe_median_val_A, spe_iqr_val_A, iou_mean_val_B, iou_median_val_B, iou_iqr_val_B, dice_mean_val_B,
                              dice_median_val_B, dice_iqr_val_B, sen_mean_val_B, sen_median_val_B, sen_iqr_val_B, spe_mean_val_B, spe_median_val_B,
                              spe_iqr_val_B):

        rows = 2
        cols = 2
        plt.figure(figsize=(cols + 20, rows + 10), dpi=320)

        plt.subplot(rows, cols, 1)
        plt.plot(epochs_count, iou_mean_train, label='Mean Train')
        plt.plot(epochs_count, iou_median_train, 'bo', label=' Median Train')
        plt.plot(epochs_count, iou_mean_val_A, color='orange', label='Mean Val A')
        plt.plot(epochs_count, iou_median_val_A, 'o', color='orange', label='Median Val A')
        plt.plot(epochs_count, iou_mean_val_B, color='red', label='Mean Val B')
        plt.plot(epochs_count, iou_median_val_B, 'ro', color='red', label='Median Val B')
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.title('IOU vs Epochs')
        plt.legend(loc="lower right")

        plt.subplot(rows, cols, 2)
        plt.plot(epochs_count, dice_mean_train, label='Mean Train')
        plt.plot(epochs_count, dice_median_train, 'bo', label=' Median Train')
        plt.plot(epochs_count, dice_mean_val_A, color='orange', label='Mean Val A')
        plt.plot(epochs_count, dice_median_val_A, 'o', color='orange', label='Median Val A')
        plt.plot(epochs_count, dice_mean_val_B, color='red', label='Mean Val B')
        plt.plot(epochs_count, dice_median_val_B, 'ro', color='red', label='Median Val B')
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('DICE')
        plt.title('DICE vs Epochs')
        plt.legend(loc="lower right")

        plt.subplot(rows, cols, 3)
        plt.plot(epochs_count, sen_mean_train, label='Mean Train')
        plt.plot(epochs_count, sen_median_train, 'bo', label=' Median Train')
        plt.plot(epochs_count, sen_mean_val_A, color='orange', label='Mean Val A')
        plt.plot(epochs_count, sen_median_val_A, 'o', color='orange', label='Median Val A')
        plt.plot(epochs_count, sen_mean_val_B, color='red', label='Mean Val B')
        plt.plot(epochs_count, sen_median_val_B, 'ro', color='red', label='Median Val B')
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity vs Epochs')
        plt.legend(loc="lower right")

        plt.subplot(rows, cols, 4)
        plt.plot(epochs_count, spe_mean_train, label='Mean Train')
        plt.plot(epochs_count, spe_median_train, 'bo', label=' Median Train')
        plt.plot(epochs_count, spe_mean_val_A, color='orange', label='Mean Val A')
        plt.plot(epochs_count, spe_median_val_A, 'o', color='orange', label='Median Val A')
        plt.plot(epochs_count, spe_mean_val_B, color='red', label='Mean Val B')
        plt.plot(epochs_count, spe_median_val_B, 'ro', color='red', label='Median Val B')
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Specificity')
        plt.title('Specificity vs Epochs')
        plt.legend(loc="lower right")

        plt.suptitle(self.model_name + '\nMAX Mean (Median) on Validation A: -> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)\n'
                                       'MAX Mean (Median) on Validation B (unseen): -> IOU: %0.1f (%0.1f), Dice: %0.1f (%0.1f)\n' % (
                         np.max(iou_mean_val_A) * 100, np.max(iou_median_val_A) * 100, np.max(dice_mean_val_A) * 100, np.max(dice_median_val_A) * 100,
                         np.max(iou_mean_val_B) * 100, np.max(iou_median_val_B) * 100, np.max(dice_mean_val_B) * 100, np.max(dice_median_val_B) * 100))

        plt.savefig("%s/1_Evaluation_Metrics.png" % (self.model_name))
        plt.close()


gan = coSegGAN()
gan.train(epochs=100, batch_size=1, sample_interval=2)