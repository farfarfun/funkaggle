import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


class MyModel:
    def __init__(self, data_root,
                 train_data_dir,
                 test_data_dir):
        self.checkpoint_path = data_root + '/models/weights.hdf5'
        self.tensorboard_path = data_root + "/logs/kaggle_deepfake-{}".format(int(time.time()))

        self.img_height, self.img_width = 150, 150

        self.training_data_dir = train_data_dir
        self.testing_data_dir = test_data_dir

        self.model = None

        self._init()

    def _init(self):
        path = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.dirname(self.tensorboard_path)
        if not os.path.exists(path):
            os.mkdir(path)

    def build(self):
        input_layer = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
        cov1 = Convolution2D(32, (3, 3),
                             name='cov1',
                             input_shape=(self.img_width, self.img_height, 3),
                             kernel_regularizer=regularizers.l2(0.0005),
                             # activity_regularizer=regularizers.l1(0.01),
                             activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(cov1)
        cov2 = Convolution2D(32, (3, 3),
                             name='cov2',
                             kernel_regularizer=regularizers.l2(0.0005),
                             # activity_regularizer=regularizers.l1(0.01),
                             activation='relu')(pool1)
        poo2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(cov2)
        cov3 = Convolution2D(64, (3, 3),
                             name='cov3',
                             kernel_regularizer=regularizers.l2(0.0005),
                             # activity_regularizer=regularizers.l1(0.01),
                             activation='relu')(poo2)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(cov3)

        fla = Flatten()(pool3)
        dense1 = Dense(64,
                       name='dense1',
                       kernel_regularizer=regularizers.l2(0.0005),
                       # activity_regularizer=regularizers.l1(0.01),
                       activation='relu', )(fla)
        drop1 = Dropout(0.5)(dense1)
        dense2 = Dense(1, name='dense2',
                       kernel_regularizer=regularizers.l2(0.0005),
                       # activity_regularizer=regularizers.l1(0.01),
                       activation='sigmoid')(drop1)

        self.model = tf.keras.Model(input_layer, dense2)
        return self.model

    def load(self):
        if os.path.exists(self.checkpoint_path):
            self.model = load_model(self.checkpoint_path)

    def train(self, batch_size=64):
        checkpoint = ModelCheckpoint(self.checkpoint_path,
                                     monitor='val_auc',
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

        self.model.fit(train_generator,
                       validation_data=validation_generator,
                       epochs=100,
                       callbacks=[tensorboard, checkpoint]
                       )

    def clear(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def predict(self, predict_dir, batch_size=64):
        test_data = ImageDataGenerator(rescale=1. / 255)
        predict_generator = test_data.flow_from_directory(
            predict_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        print(predict_generator.filenames)
        filenames = predict_generator.filenames
        result = []

        for data in tqdm(predict_generator):
            res = self.model.predict(data[0]).transpose()
            temp = np.array([res[0], data[1]]).transpose()

            result.extend(temp)
            if len(result) >= len(filenames):
                result = result[:len(filenames)]
                break

            # break
        df0 = list(np.array(result).transpose().tolist())
        df0.append(filenames[:len(result)])
        df1 = np.array(df0).transpose()
        df2 = pd.DataFrame(df1)
        df2.columns = ['label', 'real', 'name']
        df2['label'] = df2['label'].astype('float')
        df2['real'] = df2['real'].astype('float')
        df2['filename'] = df2['name'].str.extract(r'[-]([a-z]+)') + '.mp4'
        df2 = df2[['filename', 'label', 'real']]

        df3 = df2.groupby('filename').mean().reset_index()

        print(df3)

        path = '/Users/liangtaoniu/tmp/dataset/deepfake/result/submission.csv'
        df3[['filename', 'label']].to_csv(path, index=None)

        # print(result)
