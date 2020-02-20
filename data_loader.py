from __future__ import absolute_import, division, print_function, unicode_literals

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      img = image_batch[n]
      if img.shape[2] == 1:
          img = np.squeeze(img)
      ax = plt.subplot(5,5,n+1)
      plt.imshow(img, cmap='gray')
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

def get_label(file_path):
    print(file_path)
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()
  ds = ds.batch(500)
  #ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


train_data_dir  = pathlib.Path("/home/hannes/Desktop/Thesis/sacrum_classificacion_nn/Data/training")
val_data_dir    = pathlib.Path("/home/hannes/Desktop/Thesis/sacrum_classificacion_nn/Data/validation")
test_data_dir   = pathlib.Path("/home/hannes/Desktop/Thesis/sacrum_classificacion_nn/Data/testing")
image_count = len(list(train_data_dir.glob('not_sacrum/*.png')))

BATCH_SIZE = 32
IMG_HEIGHT  = 272
IMG_WIDTH   = 258
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

train_sacrum_list_ds        = tf.data.Dataset.list_files(str(train_data_dir/'sacrum/*'))
train_not_sacrum_list_ds    = tf.data.Dataset.list_files(str(train_data_dir/'not_sacrum/*'))
val_sacrum_list_ds          = tf.data.Dataset.list_files(str(val_data_dir/'sacrum/*'))
val_not_sacrum_list_ds      = tf.data.Dataset.list_files(str(val_data_dir/'not_sacrum/*'))
test_sacrum_list_ds         = tf.data.Dataset.list_files(str(test_data_dir/'sacrum/*'))
test_not_sacrum_list_ds     = tf.data.Dataset.list_files(str(test_data_dir/'not_sacrum/*'))

for f in train_sacrum_list_ds.take(5):
  print(f.numpy())

train_sacrum_labeled_ds     = train_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_not_sacrum_labeled_ds = train_not_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_sacrum_labeled_ds       = val_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_not_sacrum_labeled_ds   = val_not_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_sacrum_labeled_ds      = test_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_not_sacrum_labeled_ds  = test_not_sacrum_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_sacrum_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

in_features = np.prod(image.shape)

train_sacrum_train_ds       = prepare_for_training(train_sacrum_labeled_ds)
train_not_sacrum_train_ds   = prepare_for_training(train_not_sacrum_labeled_ds)
val_sacrum_train_ds         = prepare_for_training(val_sacrum_labeled_ds)
val_not_sacrum_train_ds     = prepare_for_training(val_not_sacrum_labeled_ds)
test_sacrum_train_ds        = prepare_for_training(test_sacrum_labeled_ds)
test_not_sacrum_train_ds    = prepare_for_training(test_not_sacrum_labeled_ds)

balanced_train_ds   = tf.data.experimental.sample_from_datasets([train_sacrum_labeled_ds, train_not_sacrum_labeled_ds], weights=[0.5, 0.5])
balanced_train_ds   = balanced_train_ds.batch(BATCH_SIZE)#.prefetch(2)
balanced_val_ds     = tf.data.experimental.sample_from_datasets([val_sacrum_labeled_ds, val_not_sacrum_labeled_ds], weights=[0.5, 0.5])
balanced_val_ds     = balanced_train_ds.batch(BATCH_SIZE)#.prefetch(2)
balanced_test_ds    = tf.data.experimental.sample_from_datasets([test_sacrum_labeled_ds, test_not_sacrum_labeled_ds], weights=[0.5, 0.5])
balanced_test_ds    = balanced_train_ds.batch(BATCH_SIZE)#.prefetch(2)


for train_imgs, train_labels in balanced_train_ds.take(1):
    train_X = train_imgs.numpy()
    train_y = train_labels.numpy()

train_y = train_y[:, 0]
print(train_y)

for val_imgs, val_labels in balanced_val_ds.take(1):
    val_X = val_imgs.numpy()
    val_y = val_labels.numpy()

val_y = val_y[:, 0]

image_batch, label_batch = next(iter(balanced_train_ds))

##########################################################################################################

EPOCHS = 100
steps_per_epoch = np.ceil(2.0*image_count/BATCH_SIZE)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(258, 272, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

#model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

print("model compiled!")

#balanced_train_ds = balanced_train_ds.make_one_shot_iterator()
#balanced_val_ds = balanced_val_ds.make_initializable_iterator()

history = model.fit(x=train_X,
                    y=train_y,
                    epochs=10,
                    steps_per_epoch=16,
                    validation_data=(val_X, val_labels),
                    validation_freq=1
                    )
    #,
    #                validation_data=balanced_val_ds,
    #                validation_steps=5)

