import os

import cv2
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(3,
                    activation='softmax'))

model = add_model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()



model.load_weights('custom_weights.hdf5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


def preprocess_image(directory):
    # Read image from directory
    img = cv2.imread(directory, cv2.IMREAD_COLOR)

    if img is not None:
        # Resize the image
        img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        # Denoise the image
        # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        # Normalize the image
        img = img / 255
        img.shape += (1,)
        return img
    else:
        pass


predicted_map = {0: 'CHART', 1: 'DIGITAL', 2: 'PHOTO'}

files = os.listdir('test_images')
for file in files:
    preprocessed_image = preprocess_image(f'test_images/{file}')
    if preprocessed_image is not None:
        preprocessed_image = preprocessed_image.reshape(-1, 256, 256, 3)
        print('Filename: ' + file + '\t' + predicted_map.get(np.argmax(model.predict(preprocessed_image), axis=1)[0]))
