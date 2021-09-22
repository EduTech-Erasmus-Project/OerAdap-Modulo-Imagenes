import os

from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as ly
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
import cv2
import numpy as np
from sklearn.utils import class_weight
from tensorflow.python.client.session import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



model = Sequential()

model.add(ly.Conv2D(16, (3, 3), activation="relu", input_shape=(128, 128, 1)))
model.add(ly.MaxPooling2D(pool_size=(2, 2)))

model.add(ly.Conv2D(32, (3, 3)))
model.add(ly.MaxPooling2D(pool_size=(2, 2)))
model.add(ly.Dropout(0.2))
model.add(ly.ReLU())


model.add(ly.Conv2D(64, (3, 3)))
model.add(ly.MaxPooling2D(pool_size=(2, 2)))
model.add(ly.Dropout(0.25))
model.add(ly.ReLU())


model.add(ly.Conv2D(128, (3, 3)))
model.add(ly.MaxPooling2D(pool_size=(2, 2)))
model.add(ly.Dropout(0.25))
model.add(ly.ReLU())


model.add(ly.Flatten())
model.add(ly.Dense(activation='relu', units=64))
model.add(ly.Dense(activation='softmax', units=3))

# Compile the CNN model, with adam optimizer.
adam = Adam()

model.load_weights('custom_weights.hdf5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


def preprocess_image(directory):
    # Read image from directory
    img = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Resize the image
        img = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        # Denoise the image
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
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
    preprocessed_image = preprocessed_image.reshape(-1, 128, 128, 1)
    print('Filename: ' + file + '\t' + predicted_map.get(np.argmax(model.predict(preprocessed_image), axis=1)[0]))
