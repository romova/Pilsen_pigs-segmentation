import numpy as np
import matplotlib.pyplot as plt
import ndnoise
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

from PIL import Image
from loguru import logger
import io3d
import io3d.datasets
logger.enable("io3d")
logger.disable("io3d")

# ------------------------------------------------------------------------------
# ---- DATA --------------------------------------------------------------------
# ------------------------------------------------------------------------------

def unzip(input_file:Path, output_dir=None):
    '''
	Otevre .zip
	param: input_file - vstupni soubor
	param: output_dir - adresar kam se maji ulozit extrahivane soubory
    '''
    from zipfile import ZipFile
    with ZipFile(input_file,"r") as zip_ref:
         for file in tqdm(
             iterable=zip_ref.namelist(),
             total=len(zip_ref.namelist()),
             desc=str(Path(input_file).name)
             ):
              zip_ref.extract(member=file, path=output_dir)
        
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
    :return: Matrix with cropped values
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

def preprocess(imgs, is_mask=False, img_rows=256, img_cols=256):
    """
    Resize and normalize input image array or binary mask for neural network input.    
    :param imgs: Input ndarray of shape [n_samples, height, width] containing grayscale images or masks
    :param is_mask: Boolean flag indicating whether the input is a segmentation mask (True) or an image (False)
    :param img_rows: Desired number of rows (height) in the output images
    :param img_cols: Desired number of columns (width) in the output images
    :return: 4D ndarray of shape [n_samples, img_rows, img_cols, 1] with normalized float32 values or binarized masks
    """
    #if not is_mask:
        #imgs = window(imgs, center=40, width=400, vmin_out=0, vmax_out=255, dtype=np.uint8)

    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = skimage.transform.resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    if is_mask:
        imgs_p = (imgs_p > 0).astype('float32')
    else:
        imgs_p = imgs_p.astype('float32') / 255.
    return imgs_p

def noisegenerator(shape):
    '''
    Prefixed use of ndnoise library
    param: shape - shape of intended matrix
    return: matrix of noise
    '''
    noise = ndnoise.generator.noises(
        shape,
        sample_spacing=[1,1,1],
        #random_generator_seed=5,
        lambda0=1,  # nastavení hrubeho sumu v pixelech
        lambda1=16, # vzdálenost shluků (rozptyl?)
        exponent=0, # prudkost rozdílu v
        method="space"
    )
    noise = (noise+5)/np.max(noise+5)
    return np.array(noise)
# ------------------------------------------------------------------------------

def image_generator_train_3DIrcad(ct_ids, promenne, batch_size=32, sz=(256, 256), axis=0):
    """
    Picks random slices from one random CT from a given dataset in one batch, along a specified axis.
    param: ct_ids -numbers of file ids intended for use
    param: batch_size=32 - number of given slices
    param: sz - intended size of output images
    param: axis - axis of view (0 = axial, 1 = coronal, 2 = sagital)
    return: matrix of slices, size (batch_size, sz)
    """
    promenne = np.array(promenne[1:])

    while True:
        # Pick random CT from available CT IDs
        dataset = '3Dircadb1'
        ct_id = np.random.choice(ct_ids, size=1)[0]        
        data3dp = io3d.datasets.read_dataset(dataset, "data3d", ct_id)                

        # Determine the number of slices along the chosen axis
        if axis == 0:
            num_slices = data3dp.data3d.shape[0]  # Axial (z-axis)
        elif axis == 1:
            num_slices = data3dp.data3d.shape[1]  # Coronal (y-axis)
        elif axis == 2:
            num_slices = data3dp.data3d.shape[2]  # Sagittal (x-axis)
        else:
            raise ValueError("Invalid axis value. Choose 0, 1, or 2.")

        # Extract random slices into a batch
        batch_slices = np.random.choice(num_slices, size=batch_size)

        # Variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        # Preprocessing - resize, window, and data type
        data3d = preprocess(
            np.moveaxis(data3dp.data3d, axis, 0),
            img_rows=sz[0], img_cols=sz[1]
        )
        
        segm3d = np.zeros(data3d.shape)                
        for i in range(len(promenne)): # scitani masek
            segm3dp = io3d.read_dataset("3Dircadb1", promenne[i], ct_id) # segmentacni maska  ###############
            segm = preprocess(
                np.moveaxis(segm3dp.data3d, axis, 0),
                is_mask=True,
                img_rows=sz[0], img_cols=sz[1]
            )
            segm3d = segm3d + segm     
        

        for slice_idx in batch_slices:
            # Preprocess the mask
            mask = segm3d[slice_idx, :, :]
            mask = np.clip(mask, 0, 1)  # Ensure the mask is binary (0 or 1)
            mask_inv = 1 - mask
            mask = np.stack((mask_inv[...,0], mask[...,0]), axis=-1)  # Stack into (H, W, 2)
            batch_y.append(mask)

            # Preprocess the raw images
            if(random.randint(0,1)>0):
                raw = (data3d[slice_idx, :, :] + noisegenerator(data3d[slice_idx, :, :].shape))/2
            else:
                raw = data3d[slice_idx, :, :]
            raw = np.stack((raw[...,0],) * 3, axis=-1)  # Stack into (H, W, 3)
            raw = np.clip(raw, 0, 1)  # Ensure the image values are within [0, 1]
            batch_x.append(raw)

        # Convert to NumPy arrays
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        yield batch_x, batch_y
        
# ------------------------------------------------------------------------------
def load_nii_as_numpy(file_path):
    """Load a NIfTI (.nii.gz) file as a NumPy array from file_path."""
    nii_img = nib.load(file_path)  # Load the NIfTI image
    data = nii_img.get_fdata()  # Get the image data as a NumPy array
    return data

def split_files(files, train_ratio=0.8):
    """Randomly split list of given files into training and validation sets based on the given ratio."""
    seed=42
    if seed is not None:
        random.seed(seed)  # Nastavení náhodného semínka
    
    random.shuffle(files)
    split_index = int(len(files) * train_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]
    return train_files, val_files


def image_generator_train_deepvesselnet(ct_ids, batch_size=32, sz=(256, 256), axis=0):
    """ DEEPVESSELNET - Picks random slices from one random CT from a given dataset in one batch, along a specified axis.    
    param: ct_ids - numbers of file ids intended for use
    param: batch_size=32 - number of given slices
    param: sz - intended size of output images
    param: axis - axis of view (0 = axial, 1 = coronal, 2 = sagital)
    return: matrix of slices, size (batch_size, sz)
    """
    while True:
        # Pick random CT from available CT IDs
        dataset = 'data/deepvesselnet'
        ct_id = ct_ids[np.random.choice(range(len(ct_ids)), size=1)[0]]
        #print(ct_id)
        
        data3dp = load_nii_as_numpy(dataset + '/raw/' + ct_id)          

        # Determine the number of slices along the chosen axis
        if axis == 0:
            num_slices = data3dp.shape[0]  # Axial (z-axis)
        elif axis == 1:
            num_slices = data3dp.shape[1]  # Coronal (y-axis)
        elif axis == 2:
            num_slices = data3dp.shape[2]  # Sagittal (x-axis)
        else:
            raise ValueError("Invalid axis value. Choose 0, 1, or 2.")

        # Extract random slices into a batch
        batch_slices = np.random.choice(num_slices, size=batch_size)

        # Variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        # Preprocessing - resize, window, and data type
        data3d = preprocess(
            np.moveaxis(data3dp, axis, 0),
            img_rows=sz[0], img_cols=sz[1]
        )
        
        segm3d = load_nii_as_numpy(dataset + '/seg/' + ct_id)*255
        #print(np.min(segm3d), np.max(segm3d), segm3d.shape)
        segm3d = preprocess(
            np.moveaxis(segm3d, axis, 0),
            is_mask=True,
            img_rows=sz[0], img_cols=sz[1]
            )
        #print(np.min(segm3d), np.max(segm3d), segm3d.shape)
        
        for slice_idx in batch_slices:
            # Preprocess the mask
            mask = segm3d[slice_idx, :, :]
            mask = np.clip(mask, 0, 1)  # Ensure the mask is binary (0 or 1)
            mask_inv = 1 - mask
            mask = np.stack((mask_inv[...,0], mask[...,0]), axis=-1)  # Stack into (H, W, 2)
            batch_y.append(mask)

            # Preprocess the raw images
            if(random.randint(0,1)>0):
                raw = (data3d[slice_idx, :, :] + noisegenerator(data3d[slice_idx, :, :].shape))/2
            else:
                raw = data3d[slice_idx, :, :]
                
            raw = np.stack((raw[...,0],) * 3, axis=-1)  # Stack into (H, W, 3)
            raw = np.clip(raw, 0, 1)  # Ensure the image values are within [0, 1]
            batch_x.append(raw)

        # Convert to NumPy arrays
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        yield batch_x, batch_y

def image_generator_train_Pigs(ct_ids, batch_size=32, sz=(256, 256), axis=0):
    """
    PILSEN_PIGS - Pick random slices from one random CT from a given dataset, along a specified axis.
    Returns inputs of shape (batch_size, H, W, 3) and masks of shape (batch_size, H, W, 2)
    param: ct_ids - list of files intended for use
    param: batch_size=32 - number of given slices
    param: sz - intended size of output images
    param: axis - axis of view (0 = axial, 1 = coronal, 2 = sagital)
    return: matrix of slices, size (batch_size, sz)
    """

    dataset = 'data/pilsen_pigs'

    while True:
        # Pick a random CT
        ct_id = ct_ids[np.random.choice(len(ct_ids))]
        data3dp = load_nii_as_numpy(f"{dataset}/{ct_id}/{ct_id}.nii.gz")

        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis value. Choose 0, 1, or 2.")
        num_slices = data3dp.shape[axis]
        batch_slices = np.random.choice(num_slices, size=batch_size)

        # Apply windowing
        data3dp1 = window(data3dp, center=40, width=350, vmin_out=0, vmax_out=255, dtype=np.uint8)
        data3dp2 = window(data3dp, center=60, width=150, vmin_out=0, vmax_out=255, dtype=np.uint8)
        data3dp3 = window(data3dp, center=110, width=110, vmin_out=0, vmax_out=255, dtype=np.uint8)

        # Rearrange and preprocess
        data3d1 = preprocess(np.moveaxis(data3dp1, axis, 0), img_rows=sz[0], img_cols=sz[1])
        data3d2 = preprocess(np.moveaxis(data3dp2, axis, 0), img_rows=sz[0], img_cols=sz[1])
        data3d3 = preprocess(np.moveaxis(data3dp3, axis, 0), img_rows=sz[0], img_cols=sz[1])

        segm3d = load_nii_as_numpy(f"{dataset}/{ct_id}/artery.nii.gz") * 255
        segm3d = preprocess(np.moveaxis(segm3d, axis, 0), is_mask=True, img_rows=sz[0], img_cols=sz[1])

        batch_x, batch_y = [], []

        for slice_idx in batch_slices:
            # Create input image with 3 channels
            if random.randint(0, 1):
                # Add noise to each channel
                weight_data = 0.8
                weight_noise = 0.2
                slice_img = np.stack([
                    (weight_data * data3d1[slice_idx] + weight_noise * noisegenerator(data3d1[slice_idx].shape)) / 2.0,
                    (weight_data * data3d2[slice_idx] + weight_noise * noisegenerator(data3d2[slice_idx].shape)) / 2.0,
                    (weight_data * data3d3[slice_idx] + weight_noise * noisegenerator(data3d3[slice_idx].shape)) / 2.0
                ], axis=-1)
            else:
                slice_img = np.stack([
                    data3d1[slice_idx],
                    data3d2[slice_idx],
                    data3d3[slice_idx]
                ], axis=-1)

            slice_img = np.clip(slice_img, 0, 1)
            batch_x.append(slice_img.astype(np.float32))

            # Prepare 2-channel mask (foreground, background)
            mask = np.clip(segm3d[slice_idx], 0, 1).astype(np.float32)
            mask_fg = mask
            mask_bg = 1.0 - mask
            mask_stacked = np.stack([mask_bg, mask_fg], axis=-1)
            batch_y.append(mask_stacked.astype(np.float32))

        # Final batches with shape:
        # batch_x: (batch_size, H, W, 3)
        # batch_y: (batch_size, H, W, 2)
        yield np.array(batch_x)[:,:,:,0,:], np.array(batch_y)[:,:,:,0,:]

# ------------------------------------------------------------------------------
# ---- MODEL -------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Weighted_binary_crossentropy loss
def weighted_binary_crossentropy(y_true, y_pred):
    '''
    param: y_pred - prediction
    param: y_true - ground true
    '''
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
    
# Dice coefficient metric
def dice_coef(y_true, y_pred, smooth=1e-6):
    '''
    param: y_pred - prediction
    param: y_true - ground true
    param: smooth - math parametr
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Additional segmentation metrics
def precision(y_true, y_pred):
    '''
    param: y_pred - prediction
    param: y_true - ground true
    '''
    y_pred_bin = K.round(y_pred)
    true_positives = K.sum(y_true * y_pred_bin)
    predicted_positives = K.sum(y_pred_bin)
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    '''
    param: y_pred - prediction
    param: y_true - ground true
    '''
    y_pred_bin = K.round(y_pred)
    true_positives = K.sum(y_true * y_pred_bin)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    '''
    param: y_pred - prediction
    param: y_true - ground true
    '''
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + K.epsilon())











