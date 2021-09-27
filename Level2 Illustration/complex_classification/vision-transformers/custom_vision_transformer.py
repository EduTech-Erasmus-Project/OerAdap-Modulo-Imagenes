import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
from keras.utils.data_utils import Sequence

from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

image_size = 128
batch_size = 16
n_classes = 3


def read_data():
    return pd.read_csv(
        '/media/edutech-pc06/Elements1/DataSet/ClasificacionPorContenido/PhotoChartDigital/dataframe.csv')


df_train = read_data()
df_train = shuffle(read_data())
x_train, x_test, y_train, y_test = train_test_split(df_train['path'].to_list(), df_train['class'].to_list(),
                                                    test_size=0.30,
                                                    random_state=42)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = {i: class_weights[i] for i in range(3)}


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
        img = cv2.imread(directory)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            # Resize the image
            img = cv2.resize(src=img, dsize=(image_size, image_size))
            # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
            # Denoise the image
            # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            # Normalize the image
            img = img / 255
            return img
        else:
            pass


my_training_batch_generator = Data_Generator(x_train, y_train, batch_size)

my_validation_batch_generator = Data_Generator(x_test, y_test, batch_size)

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 1

patch_size = 7  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [56, 28]  # Size of the dense layers of the final classifier


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation = tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x


class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = L.Dense(units = projection_dim)
        self.position_embedding = L.Embedding(
            input_dim=num_patches, output_dim = projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def vision_transformer():
    inputs = L.Input(shape=(image_size, image_size, 1))

    # Create patches.
    patches = Patches(patch_size)(inputs)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = L.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = L.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = L.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = L.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = L.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = L.Flatten()(representation)
    representation = L.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Classify outputs.
    logits = L.Dense(n_classes, activation='softmax')(features)

    # Create the model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model


initial_learning_rate = learning_rate

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = vision_transformer()

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 min_delta=1e-4,
                                                 patience=5,
                                                 mode='max',
                                                 restore_best_weights=True,
                                                 verbose=1)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./model.hdf5',
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  mode='max')

callbacks = [earlystopping, checkpointer]

model.summary()

model.fit(x=my_training_batch_generator,
          validation_data=my_validation_batch_generator,
          epochs=num_epochs,
          callbacks=callbacks, class_weight=class_weights)

print('Training results')
print(model.evaluate(my_training_batch_generator))

print('Validation results')
print(model.evaluate(my_validation_batch_generator))
