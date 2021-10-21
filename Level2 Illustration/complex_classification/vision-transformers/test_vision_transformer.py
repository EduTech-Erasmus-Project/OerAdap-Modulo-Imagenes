import os

import cv2
import numpy
import pandas
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from keras.utils.data_utils import Sequence
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from vit_keras import vit, utils
from keras.models import load_model

IMAGE_SIZE = 128

model = load_model('model.h5')

url = 'https://ak.picdn.net/shutterstock/videos/23133964/thumb/1.jpg'
image = utils.read(url, 128)
X = vit.preprocess_inputs(image).reshape(1, 128, 128, 3)
y = model.predict(X)
print(y[0])

files = os.listdir('test_images')
for file in files:
    try:
        image = utils.read(f'test_images/{file}', 128)
        X = vit.preprocess_inputs(image).reshape(1, 128, 128, 3)
        y = model.predict(X)
        print('Archivo: ', file, '\t', numpy.argmax(y[0]))
    except Exception as e:
        print(e)
