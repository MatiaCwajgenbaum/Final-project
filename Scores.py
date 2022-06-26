import numpy as np
import keras.backend as K

def IoU_coeff(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    mask = K.sum(y_true_f) + K.sum(y_pred_f)
    union = mask - K.sum(intersection)
    iou = (K.sum(intersection) + smooth) / (union + smooth)
    return iou

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, num_labels, flag=0):
    dice = 0
    for index in range(num_labels):
        if flag:
            print(dice_coef(y_true[:, :, index], y_pred[:, :, index]))
        dice += dice_coef(y_true[:, :, index], y_pred[:, :, index])
    return dice / num_labels  # taking average


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
