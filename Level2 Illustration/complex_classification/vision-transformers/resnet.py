import cv2
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras.optimizer_v2.adam import Adam
from keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight


def read_data():
    return pd.read_csv(
        'dataframe.csv')


dataframe = shuffle(read_data())
x_train, x_test, y_train, y_test = train_test_split(dataframe['path'].to_list(), dataframe['class'].to_list(),
                                                    test_size=0.30,
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
            # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
            # Denoise the image
            # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            # Normalize the image
            img = img / 255
            return img
        else:
            pass


# Preprocess a single image and return an array.

batch_size = 64
my_training_batch_generator = Data_Generator(x_train, y_train, batch_size)

my_validation_batch_generator = Data_Generator(x_test, y_test, batch_size)

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.35))
add_model.add(Dense(3,
                    activation='softmax'))

model = add_model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.005),
              metrics=['accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001)
early_stop = EarlyStopping(monitor='loss', patience=2)

history = model.fit_generator(my_training_batch_generator,
                              epochs=5,
                              verbose=True,
                              validation_data=my_validation_batch_generator, class_weight=class_weights,
                              callbacks=[reduce_lr, early_stop])

model.save_weights('custom_weights.hdf5')

accuracy_training = history.history['accuracy']
accuracy_testing = history.history['val_accuracy']

acc_training = np.array(accuracy_training)
acc_testing = np.array(accuracy_testing)

dataframe_accuracies = pd.DataFrame(list(zip(acc_training, acc_testing)), columns=['ACC_Training', 'ACC_Testing'])
dataframe_accuracies.to_csv('dataframe_accuracies.csv')
dataframe_accuracies.head()
