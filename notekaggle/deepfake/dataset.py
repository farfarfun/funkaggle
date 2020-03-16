import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

checkpoint_path = 'models/weights.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_acc',
                             verbose=1,
                             mode='max')
tensorboard_path = "kaggle_deepfake-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(tensorboard_path),
                          update_freq=10,
                          write_graph=True,
                          write_images=True,
                          profile_batch=0
                          )

batch_size = 16
img_height, img_width = 600, 800
training_data_dir = "/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_0_img/"
# testing_data_dir = "/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_0_valid/"
testing_data_dir = "/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_1_img/"

train_data = ImageDataGenerator(rescale=1. / 255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data.flow_from_directory(
    training_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_data.flow_from_directory(
    testing_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


def build2():
    input_layer = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    cov1 = Convolution2D(32, (3, 3),
                         name='cov1',
                         input_shape=(img_width, img_height, 3),
                         activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(cov1)
    cov2 = Convolution2D(32, (3, 3),
                         name='cov2',
                         activation='relu')(pool1)
    poo2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(cov2)
    cov3 = Convolution2D(64, (3, 3),
                         name='cov3',
                         activation='relu')(poo2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(cov3)

    fla = Flatten()(pool3)
    dense1 = Dense(64, activation='relu', name='dense1')(fla)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation='sigmoid', name='dense2')(drop1)

    return tf.keras.Model(input_layer, dense2)


model = build2()

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy", keras.metrics.AUC()])

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=100,
          callbacks=[tensorboard, checkpoint]
          )

for data in validation_generator:
    res = model.predict(data[0]).transpose()
    temp = np.array([res[0], data[1]]).transpose()
    temp = np.round(temp, 3)

    print(temp)

    break
