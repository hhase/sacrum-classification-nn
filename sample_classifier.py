
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
import matplotlib.pyplot as plt

#Initialising CNN
classifier = Sequential()

#Adding Convolution Layer 1

classifier.add(Convolution2D(32,3,3,input_shape=(258,272,3),activation='relu'))
#Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#Adding Convolution Layer 2

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

#Flatten
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
#classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the model to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen= ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Data/training',
                                                    target_size=(258,272),
                                                    batch_size=128,
                                                    class_mode='binary')

label_map = (training_set.class_indices)
print(label_map)

test_set = test_datagen.flow_from_directory('Data/validation',
                                                        target_size=(258,272),
                                                        batch_size=32,
                                                        class_mode='binary')

from collections import Counter

counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

from keras import callbacks
log_dir = "./logs"
tensorboard = callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=0,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=False,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None,
                                     embeddings_data=None,
                                     update_freq='epoch')


history = classifier.fit_generator(training_set,
                        steps_per_epoch=2,#0,#8000,
                        epochs=5, #25
                        class_weight=class_weights,
                        validation_data=test_set,
                        validation_steps=20,#00,
                        workers=4,
                        callbacks=[tensorboard]
                        )


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()