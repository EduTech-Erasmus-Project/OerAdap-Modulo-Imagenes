import os

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.core.protobuf.config_pb2 import ConfigProto
import tensorflow.keras.layers as ly
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import cv2
import numpy as np
from tensorflow.python.client.session import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.25))
add_model.add(Dense(3,
                    activation='softmax'))

model = add_model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0035),
              metrics=['accuracy'])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.load_weights('/home/edutech-pc06/PycharmProjects/OerAdap-Modulo-Imagenes/Level2 Illustration/Level3 Digital/custom_weights.hdf5')


def preprocess_image(directory):
    # Read image from directory
    img = cv2.imread(directory, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        img = img / 255
        return img
    else:
        pass


predicted_map = {0: 'Animated illustration', 1: 'Logo', 2: 'Screenshot'}

files = os.listdir('/home/edutech-pc06/PycharmProjects/OerAdap-Modulo-Imagenes/Level2 Illustration/Level3 Digital/test_images')
for file in files:
    preprocessed_image = preprocess_image(f'/home/edutech-pc06/PycharmProjects/OerAdap-Modulo-Imagenes/Level2 Illustration/Level3 Digital/test_images/{file}')
    preprocessed_image = preprocessed_image.reshape(-1, 256, 256, 3)
    print('Filename: ' + file + '\t' + predicted_map.get(np.argmax(model.predict(preprocessed_image), axis=1)[0]))
