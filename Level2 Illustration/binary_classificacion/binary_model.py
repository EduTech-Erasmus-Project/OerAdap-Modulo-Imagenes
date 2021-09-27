# /home/edutech-pc06/PycharmProjects/OerAdap-Modulo-Imagenes/Level2 Illustration/binary_classificacion/dataset.csv

# %%

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

# %%

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
    return pd.read_csv(
        '/home/edutech-pc06/PycharmProjects/OerAdap-Modulo-Imagenes/Level2 Illustration/binary_classificacion/dataset.csv')


dataframe = shuffle(read_data())
x_train, x_test, y_train, y_test = train_test_split(dataframe['path'].to_list(), dataframe['class'].to_list(),
                                                    test_size=0.30,
                                                    random_state=42)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = {i: class_weights[i] for i in range(2)}

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
        img = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize the image
            img = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
            gauss = cv2.GaussianBlur(img, (5, 5), 0)
            canny = cv2.Canny(gauss, 50, 150)
            cont, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            img = np.zeros(img.shape, dtype='uint8')
            img.fill(255)
            cv2.drawContours(img, cont, -1, (0, 0, 255), 2)

            img = img / 255
            img.shape += (1,)
            return img
        else:
            pass


# Preprocess a single image and return an array.

batch_size = 32
my_training_batch_generator = Data_Generator(x_train, y_train, batch_size)

my_validation_batch_generator = Data_Generator(x_test, y_test, batch_size)


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
model.add(ly.Dense(activation='sigmoid', units=1))

# Compile the CNN model, with adam optimizer.
adam = Adam()
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

json_model = model.to_json()

with open('custom_json_model.json', 'w') as json_file:
    json_file.write(json_model)

history = model.fit_generator(
    generator=my_training_batch_generator,
    epochs=2,
    verbose=1,
    validation_data=my_validation_batch_generator,
    use_multiprocessing=True,
    class_weight=class_weights,
)

model.save_weights('custom_weights.hdf5')

accuracy_training = history.history['accuracy']
accuracy_testing = history.history['val_accuracy']
epochs = 2

acc_training = np.array(accuracy_training)
acc_testing = np.array(accuracy_testing)

dataframe_accuracies = pd.DataFrame(list(zip(acc_training, acc_testing)), columns=['ACC_Training', 'ACC_Testing'])
dataframe_accuracies.to_csv('dataframe_accuracies.csv')
dataframe_accuracies.head()
