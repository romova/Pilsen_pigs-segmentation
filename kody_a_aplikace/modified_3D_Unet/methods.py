import numpy as np
import random
from numbers import Number
from typing import Optional
from pathlib import Path
import scipy
import cv2
import skimage.transform
import nibabel as nib
import h5py
import json
from tqdm import tqdm

import sys
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, mixed_precision
#from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Activation, Concatenate, BatchNormalization
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Dropout, Activation, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import HeNormal


def load_nii_as_numpy(file_path):
    """Load a NIfTI (.nii.gz) file as a NumPy array."""
    nii_img = nib.load(file_path)  # Load the NIfTI image
    data = nii_img.get_fdata()  # Get the image data as a NumPy array
    return data
    
def resample_to_voxelsize(img_np, original_spacing, target_spacing=(1.0, 1.0, 1.0), order=3):
    """
    Resample a 3D image to a specified voxel spacing using interpolation.
    :param img_np: Input 3D ndarray representing the image volume
    :param original_spacing: Tuple or list of floats (z, y, x), representing the spacing of the input volume
    :param target_spacing: Tuple of floats (z, y, x), desired voxel spacing for output volume
    :param order: Interpolation order (0=nearest, 1=linear, 3=cubic, etc.)
    :return: 3D ndarray resampled to the target voxel spacing
    """
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    return scipy.ndimage.zoom(img_np, zoom_factors, order=order)


def sliding_window_inference(nifti_path, model: Model, window_shape=(256, 256, 64), stride=(128, 128, 32)):
    """
    Apply a trained 3D model using sliding window inference on a NIfTI volume after preprocessing.

    Parameters:
        nifti_path (str): path to the CT scan (file name).
        model (Model): Trained Keras model.
        window_shape (tuple): Shape of the sliding window.
        stride (tuple): Step size of sliding window.
    
    Returns:
        output_volume (np.ndarray): Prediction mask for the entire volume.
    """
    nifti_path = Path(nifti_path)
    
    # === 1. Load NIfTI file ===
    raw_nii = nib.load(nifti_path)
    raw = raw_nii.get_fdata()
    spacing = raw_nii.header.get_zooms()[:3]  # (z, y, x)

    # === 2. Apply windowing (CT intensity range) ===
    raw = window(raw, center=40, width=400)
    bones = window(raw, center=270, width=310)

    # === 3. Resample to isotropic voxels ===
    raw = resample_to_voxelsize(raw, spacing, target_spacing=(1.0, 1.0, 1.0), order=3)
    bones = resample_to_voxelsize(bones, spacing, target_spacing=(1.0, 1.0, 1.0), order=3)
    
    # === 4. Normalize and expand to 3 channels ===
    raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-5)
    raw = np.stack([raw] * 3, axis=-1)  # From (Z, Y, X) to (Z, Y, X, 3)

    original_shape = raw.shape[:-1]

    # === 5. Padding ===
    pad_width = [
        (0, max(0, window_shape[0] - original_shape[0] % stride[0])),
        (0, max(0, window_shape[1] - original_shape[1] % stride[1])),
        (0, max(0, window_shape[2] - original_shape[2] % stride[2])),
        (0, 0)
    ]
    volume_padded = np.pad(raw, pad_width, mode='constant')
    padded_shape = volume_padded.shape

    output = np.zeros(padded_shape[:-1], dtype=np.float32)
    count_map = np.zeros(padded_shape[:-1], dtype=np.float32)

    # === 6. Sliding window inference ===
    for z in tqdm(range(0, padded_shape[0] - window_shape[0] + 1, stride[0])):
        for y in range(0, padded_shape[1] - window_shape[1] + 1, stride[1]):
            for x in range(0, padded_shape[2] - window_shape[2] + 1, stride[2]):
                patch = volume_padded[z:z+window_shape[0],
                                      y:y+window_shape[1],
                                      x:x+window_shape[2], :]

                patch_input = np.expand_dims(patch, axis=0)  # Batch dim
                prediction = model.predict(patch_input, verbose=0)[0]

                output[z:z+window_shape[0],
                       y:y+window_shape[1],
                       x:x+window_shape[2]] += prediction[..., 0]
                count_map[z:z+window_shape[0],
                          y:y+window_shape[1],
                          x:x+window_shape[2]] += 1

    output /= np.maximum(count_map, 1e-5)
    output = output[:original_shape[0], :original_shape[1], :original_shape[2]]
    output = (output > 0.2).astype(np.float32)
    bones = (bones/np.max(bones) > 0.5).astype(np.float32)
    

    return output-bones

def window(
    data3d: np.ndarray,
    vmin: Optional[Number] = None,
    vmax: Optional[Number] = None,
    center: Optional[Number] = None,
    width: Optional[Number] = None,
    vmin_out: Optional[Number] = 0,
    vmax_out: Optional[Number] = 255,
    dtype=np.uint8,
):
    """
    Rescale input ndarray and trim the outlayers. Used for image intensity windowing.
    :param data3d: ndarray with numbers
    :param vmin: minimal input value. Skipped if center and width is given.
    :param vmax: maximal input value. Skipped if center and width is given.
    :param center: Window center
    :param width: Window width
    :param vmin_out: Output mapping minimal value
    :param vmax_out: Output mapping maximal value
    :param dtype: Output dtype
    :return:
    """
    if width and center:
        vmin = center - (width / 2.0)
        vmax = center + (width / 2.0)

    # logger.debug(f"vmin={vmin}, vmax={vmax}")
    k = float(vmax_out - vmin_out) / (vmax - vmin)
    q = vmax_out - k * vmax
    # logger.debug(f"k={k}, q={q}")
    data3d_out = data3d * k + q

    data3d_out[data3d_out > vmax_out] = vmax_out
    data3d_out[data3d_out < vmin_out] = vmin_out

    return data3d_out.astype(dtype)

def conv_block(x, filters, n_convs, kernel_initializer, dropout_rate):
    """
    Creates a convolutional block with a specified number of Conv3D layers and dropout.

    :param x: Input tensor to the block
    :param filters: Number of filters for each convolutional layer
    :param n_convs: Number of convolutional layers in the block
    :param kernel_initializer: Initializer for convolutional kernels (e.g., HeNormal)
    :param dropout_rate: Dropout rate applied after convolutions
    :return: Output tensor after convolutions and dropout
    """
    for _ in range(n_convs):
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def get_3d_unet(shape=(128, 128, 128), n_convs_per_block=4):
    """
    Builds a 3D U-Net architecture for volumetric image segmentation.

    :param shape: Tuple (depth, height, width), shape of the input volume (excluding channels)
    :param n_convs_per_block: Number of Conv3D layers in each encoder/decoder block
    :return: Compiled Keras Model representing the 3D U-Net
    """
    tf.keras.backend.set_image_data_format('channels_last')
    seed = 42
    tf.random.set_seed(seed)
    init = HeNormal(seed=seed)
    inputs = layers.Input(shape=(*shape, 3))

    # Encoder
    filters = [8, 16, 32, 64, 128]
    encoders = []
    x = inputs
    for f in filters:
        x = conv_block(x, f, n_convs_per_block, init, dropout_rate=0.2)
        encoders.append(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)

    # Bottleneck
    x = conv_block(x, 256, n_convs_per_block, init, dropout_rate=0.3)

    # Decoder
    for i, f in reversed(list(enumerate(filters))):
        x = layers.Conv3DTranspose(f, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)
        x = layers.concatenate([x, encoders[i]])
        x = conv_block(x, f, n_convs_per_block, init, dropout_rate=0.2)

    outputs = layers.Conv3D(2, (1, 1, 1), activation='sigmoid')(x)
    return Model(inputs, outputs)


# Metrics
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient, a measure of overlap between two samples.
    
    Dice coefficient is commonly used as a metric for evaluating segmentation performance.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted binary mask (tensor)
    :param smooth: Smoothing factor to avoid division by zero
    :return: Dice coefficient (float)
    """
    y_true_f = K.cast(K.flatten(y_true), dtype=K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), dtype=K.floatx())
    smooth = K.cast(smooth, dtype=K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def precision(y_true, y_pred):
    """
    Computes the precision (positive predictive value) for binary classification.
    
    Precision = TP / (TP + FP)
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted probability or binary mask (tensor)
    :return: Precision score (float)
    """
    y_pred_bin = K.round(y_pred)
    true_positives = K.sum(y_true * y_pred_bin)
    predicted_positives = K.sum(y_pred_bin)
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """
    Computes the recall (sensitivity, true positive rate) for binary classification.
    
    Recall = TP / (TP + FN)
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted probability or binary mask (tensor)
    :return: Recall score (float)
    """
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    y_pred_bin = K.round(y_pred)
    true_positives = K.sum(y_true * y_pred_bin)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    """
    Computes the F1-score, the harmonic mean of precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted probability or binary mask (tensor)
    :return: F1 score (float)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + K.epsilon())


from tensorflow.keras import backend as K

def dice_coef_loss(y_true, y_pred, smooth=1):
    """
    Computes the Dice coefficient loss, which is 1 - Dice coefficient.
    
    Dice coefficient is a similarity measure, and the Dice loss penalizes the 
    dissimilarity between predicted and true binary masks.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted binary mask (tensor)
    :param smooth: Smoothing factor to avoid division by zero
    :return: Dice coefficient loss (float)
    """
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    smooth = K.cast(smooth, dtype=K.floatx())
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice


def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=0.75, smooth=1):
    """
    Computes the Focal Tversky loss, which emphasizes hard-to-classify pixels.
    
    This loss is effective for handling imbalanced classes in segmentation tasks.
    It is a generalization of the Tversky index and is often used in medical image segmentation.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted binary mask (tensor)
    :param alpha: Weighting factor for false positives
    :param beta: Weighting factor for false negatives
    :param gamma: Focusing parameter for the loss
    :param smooth: Smoothing factor to avoid division by zero
    :return: Focal Tversky loss (float)
    """
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    smooth = K.cast(smooth, dtype=K.floatx())
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return K.pow((1 - tversky_index), gamma)


def tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, smooth=1):
    """
    Computes the Tversky loss, which is particularly useful for imbalanced segmentation tasks.
    
    The Tversky loss emphasizes false positives and false negatives by introducing
    weighting factors alpha and beta to control their importance.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted binary mask (tensor)
    :param alpha: Weighting factor for false positives
    :param beta: Weighting factor for false negatives
    :param smooth: Smoothing factor to avoid division by zero
    :return: Tversky loss (float)
    """
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    smooth = K.cast(smooth, dtype=K.floatx())
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    
    return 1 - ((TP + smooth) / (TP + alpha * FP + beta * FN + smooth))


def jaccard_loss(y_true, y_pred, smooth=1):
    """
    Computes the Jaccard loss, a measure of overlap between two sets.
    
    The Jaccard index is calculated as the intersection over the union of two sets,
    and the loss is 1 - Jaccard index.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted binary mask (tensor)
    :param smooth: Smoothing factor to avoid division by zero
    :return: Jaccard loss (float)
    """
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    smooth = K.cast(smooth, dtype=K.floatx())
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1 - ((intersection + smooth) / (union + smooth))


def weighted_binary_crossentropy(y_true, y_pred):
    """
    Computes the weighted binary crossentropy loss.
    
    This loss function assigns different weights to the positive and negative class, 
    allowing to handle imbalanced data more effectively.
    
    :param y_true: Ground truth binary mask (tensor)
    :param y_pred: Predicted probability (tensor)
    :return: Weighted binary crossentropy loss (float)
    """
    # Default weights (used if not explicitly set)
    default_weights = [1.0, 1.0]  # Default: equal weight for both classes
    weights = getattr(weighted_binary_crossentropy, "weights", default_weights)

    weight_positive, weight_negative = weights

    # Convert to float tensors
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())

    # Avoid log(0)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    # Compute weighted binary crossentropy loss
    loss = - (weight_positive * y_true * K.log(y_pred) + 
              weight_negative * (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(loss)


