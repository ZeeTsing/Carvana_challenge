import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import downscale_local_mean
from os.path import join
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

input_folder = 'G:/Github/Caravan_challenge/data'
print(input_folder)

df_mask = pd.read_csv(join(input_folder, 'train_masks.csv'), usecols=['img'])
ids_train = df_mask['img'].map(lambda s: s.split('_')[0]).unique()

imgs_idx = list(range(1, 17))

load_img = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))
load_mask = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
resize = lambda im: downscale_local_mean(im, (4,4) if im.ndim==2 else (4,4,1))
mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

num_train = 32  # len(ids_train)

# Load data for position id=1
X = np.empty((num_train, 320, 480, 9), dtype=np.float32)
y = np.empty((num_train, 320, 480, 1), dtype=np.float32)


idx = 1 # Rotation index
for i, img_id in enumerate(ids_train[:num_train]):
    imgs_id = [resize(load_img(img_id, j)) for j in imgs_idx]
    # Input is image + mean image per channel + std image per channel
    X[i, ..., :9] = np.concatenate([imgs_id[idx-1], np.mean(imgs_id, axis=0), np.std(imgs_id, axis=0)], axis=2)
    y[i] = resize(np.expand_dims(load_mask(img_id, idx), 2)) / 255.
    if i % 5 ==0:
        print('processed {} images'.format(i))
      
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize input and output
X_mean = X_train.mean(axis=(0,1,2), keepdims=True)
X_std = X_train.std(axis=(0,1,2), keepdims=True)

X_train -= X_mean
X_train /= X_std

X_val -= X_mean
X_val /= X_std

# Now let's use Tensorflow to write our own dice_coeficcient metric
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Sequential

model = Sequential()
model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(320, 480, 9)) )
model.add( Conv2D(32, 3, activation='relu', padding='same') )
model.add( Conv2D(1, 5, activation='sigmoid', padding='same') )

model.compile(optimizer = 'Adam', loss = BinaryCrossentropy(), metrics=['accuracy',dice_coef])
print(model.summary())

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "G:/Github/Caravan_challenge/training_tf/naive_org/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,
    save_freq='epoch')

model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=5, verbose=2)

tf.keras.models.save_model(model,'G:/Github/Caravan_challenge/training_tf/naive_1')