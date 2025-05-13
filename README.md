!pip install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(zip_path))

base_dir = os.path.join(pathlib.Path(zip_path).parent, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator
)

model_json = model.to_json()
with open("model_gats_gossos.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_gats_gossos.weights.h5")

from google.colab import files
files.download("model_gats_gossos.json")
files.download("model_gats_gossos.weights.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
