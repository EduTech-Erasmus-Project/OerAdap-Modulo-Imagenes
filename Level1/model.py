#%%
import itertools

import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as ly
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
import cv2
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

#%%

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def read_data():
    return pd.read_csv('dataframe.csv')


dataframe = shuffle(read_data())
x_train, x_test, y_train, y_test = train_test_split(dataframe['path'].to_list(), dataframe['class'].to_list(),
                                                    test_size=0.33,
                                                    random_state=42)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = {i: class_weights[i] for i in range(3)}

print(f'Class weights: {class_weights}')


# This class inherit Sequence class in order to create a custom generator
class Data_Generator(Sequence):

    # We feed oun generator with our parameters.
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    # Computes the number of batches to produce.
    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    # We preprocess our dataset with the current batch (Here is where magic happens).
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        flag = False
        for directory in batch_x:
            if cv2.imread(directory) is None:
                flag = True

        if not flag:
            return np.array(
                [self.preprocess_image(directory) for directory in batch_x]
            ), np.array(batch_y)

    # Preprocess a single image and return an array.
    def preprocess_image(self, directory):
        # Read image from directory
        img = cv2.imread(directory, cv2.IMREAD_COLOR)

        if img is not None:
            # Resize the image
            img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            # Denoise the image
            #img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            # Normalize the image
            img = img / 255
            #img.shape += (1,)
            return img
        else:
            pass


# Preprocess a single image and return an array.

batch_size = 10
my_training_batch_generator = Data_Generator(x_train, y_train, batch_size)

my_validation_batch_generator = Data_Generator(x_test, y_test, batch_size)

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

history = model.fit_generator(my_training_batch_generator,
                              epochs=2,
                              verbose=True,
                              validation_data=my_validation_batch_generator, class_weight=class_weights)

json_model = model.to_json()

with open('custom_json_model.json', 'w') as json_file:
    json_file.write(json_model)

model.save_weights('custom_weights.hdf5')

accuracy_training = history.history['accuracy']
accuracy_testing = history.history['val_accuracy']

acc_training = np.array(accuracy_training)
acc_testing = np.array(accuracy_testing)

dataframe_accuracies = pd.DataFrame(list(zip(acc_training, acc_testing)), columns=['ACC_Training', 'ACC_Testing'])
dataframe_accuracies.to_csv('dataframe_accuracies.csv')
dataframe_accuracies.head()

#
# def preprocess_image(directory):
#     # Read image from directory
#     img = cv2.imread(directory, cv2.IMREAD_COLOR)
#
#     if img is not None:
#         # Resize the image
#         img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
#         # Denoise the image
#         # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
#         # Normalize the image
#         img = img / 255
#         # img.shape += (1,)
#         return img
#     else:
#         pass
#
#
# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
#
#
# x_test_chunks = list(chunks(x_test, 100))
# print(len(x_test_chunks))
# predictions: list = list()
#
# test_index: int = -1
# y_test3: list = []
# for index1, chunk in enumerate(x_test_chunks):
#     images = []
#     if index1 > 200: break
#     for index, file in enumerate(chunk):
#         preprocessed_image = preprocess_image(file)
#         if preprocessed_image is not None:
#             test_index += 1
#             preprocessed_image = preprocessed_image.reshape(-1, 256, 256, 3)
#             images.append(preprocessed_image)
#             y_test3.append(y_test[test_index])
#         else:
#             test_index += 1
#     images = np.vstack(images)
#     ypred = model.predict(images)
#     ypred = ypred.argmax(axis=-1)
#     predictions.append(ypred)
#     print(f'Chunk number {index1} finished...')
#
#
# merged_list = list(itertools.chain(*predictions))
#
# target = ['Formula', 'Illustracion', 'Tabla']
# fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
#
#
# def multiclass_roc_auc_score(y_test, y_pred, average='macro'):
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     print(y_test)
#     y_pred = lb.transform(y_pred)
#
#     for idx, c_label in enumerate(target):
#         y_test[:, idx].astype(int)
#
#         fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
#         c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
#     c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
#     return roc_auc_score(y_test, y_pred, average=average)
#
#
# print('ROC AUC score: ', multiclass_roc_auc_score(y_test3, merged_list))
#
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# plt.show()
#
