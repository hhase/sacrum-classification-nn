from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

train_data_dir  = pathlib.Path("/home/hannes/Desktop/Thesis/sacrum_classificacion_nn/Data/training")
data_dir = pathlib.Path(train_data_dir)

image_count = len(list(train_data_dir.glob('not_sacrum/*.png')))
print(image_count)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(258, 272, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
