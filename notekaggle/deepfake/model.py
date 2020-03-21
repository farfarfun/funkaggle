import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class MyModel:
    def __init__(self, train_data_dir=None, test_data_dir=None, data_root='/Users/liangtaoniu/tmp/dataset/deepfake'):
        self.checkpoint_path = data_root + '/models/weights.hdf5'
        self.tensorboard_path = data_root + "/logs/kaggle_deepfake-{}".format(int(time.time()))

        self.img_height, self.img_width = 150, 150

        self.training_data_dir = train_data_dir or "/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_1_img/"

        self.testing_data_dir = test_data_dir or "/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_1_img/"

        self.model = None

    def build2(self):
        input_layer = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
        cov1 = Convolution2D(32, (3, 3),
                             name='cov1',
                             input_shape=(self.img_width, self.img_height, 3),
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

        self.model = tf.keras.Model(input_layer, dense2)
        return self.model

    def load(self):
        if os.path.exists(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)

    def train(self, batch_size=16):
        checkpoint = ModelCheckpoint(self.checkpoint_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     mode='max')

        tensorboard = TensorBoard(log_dir=self.tensorboard_path,
                                  update_freq=10,
                                  write_graph=True,
                                  write_images=True,
                                  profile_batch=0
                                  )

        train_data = ImageDataGenerator(rescale=1. / 255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

        test_data = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data.flow_from_directory(
            self.training_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_data.flow_from_directory(
            self.testing_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary')

        self.model.compile(loss="binary_crossentropy",
                           optimizer="rmsprop",
                           metrics=["accuracy", keras.metrics.AUC()])

        self.load()
        self.model.fit(train_generator,
                       validation_data=validation_generator,
                       epochs=100,
                       callbacks=[tensorboard, checkpoint]
                       )

    def predict(self, batch_size=16):
        test_data = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_data.flow_from_directory(
            self.testing_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary')

        for data in validation_generator:
            res = self.model.predict(data[0]).transpose()
            temp = np.array([res[0], data[1]]).transpose()
            temp = np.round(temp, 3)

            print(temp)

            break


model = MyModel()
model.build2()
model.train()
