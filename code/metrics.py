'''
Code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''

from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf

def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)

        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise

        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise

        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss

def dice_coef_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def loss_structure(y_true,y_pred):
    encoded_img_A = y_pred[:, :, :, :, 0]
    encoded_fake_B = y_pred[:, :, :, :, 1]
    encoded_img_B = y_pred[:, :, :, :, 2]
    encoded_fake_A = y_pred[:, :, :, :, 3]

    mse1 = tf.keras.losses.MAE(encoded_img_A, encoded_fake_B)
    mse2 = tf.keras.losses.MAE(encoded_img_B, encoded_fake_A)

    return 0.5*(mse1 + mse2)





