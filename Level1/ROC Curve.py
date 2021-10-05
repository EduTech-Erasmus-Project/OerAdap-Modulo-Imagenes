import itertools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import tensorflow.keras.layers as ly
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
import cv2
import numpy as np


def read_data():
    return pd.read_csv('/media/edutech-pc06/Elements1/DataSet/ClasificacionPorContenido/dataframe.csv')


dataframe = shuffle(read_data())
x_train, x_test, y_train, y_test = train_test_split(dataframe['path'].to_list(), dataframe['class'].to_list(),
                                                    test_size=0.99,
                                                    random_state=42)


def get_model():
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

    return model


def preprocess_image(directory):
    # Read image from directory
    img = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Resize the image
        img = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        # Denoise the image
        # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        # Normalize the image
        img = img / 255
        img.shape += (1,)
        return img
    else:
        pass


model = get_model()
predicted_map = {0: 'FORMULA', 1: 'ILUSTRACION', 2: 'TABLA'}


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


x_test_chunks = list(chunks(x_test, 100))
print(len(x_test_chunks))
predictions: list = list()

test_index: int = -1
y_test3: list = []
for index1, chunk in enumerate(x_test_chunks):
    images = []

    for index, file in enumerate(chunk):
        preprocessed_image = preprocess_image(file)
        if preprocessed_image is not None:
            test_index += 1
            preprocessed_image = preprocessed_image.reshape(-1, 128, 128, 1)
            images.append(preprocessed_image)
            y_test3.append(y_test[test_index])
        else:
            test_index += 1
    images = np.vstack(images)
    ypred = model.predict(images)
    ypred = ypred.argmax(axis=-1)
    predictions.append(ypred)
    print(f'Chunk number {index1} finished...')


merged_list = list(itertools.chain(*predictions))

target = ['Formula', 'Illustracion', 'Tabla']
fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))


def multiclass_roc_auc_score(y_test, y_pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    print(y_test)
    y_pred = lb.transform(y_pred)

    for idx, c_label in enumerate(target):
        y_test[:, idx].astype(int)

        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)


print('ROC AUC score: ', multiclass_roc_auc_score(y_test3, merged_list))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()
