"""DFDC deepfake 检测使用的简单三层卷积二分类模型。"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from farlog import getLogger
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

logger = getLogger("funkaggle")


class MyModel:
    """基于 tf.keras 的三层卷积二分类模型，用于区分真实/伪造人脸帧。

    Args:
        data_root: 数据与权重根目录；权重存放于 `<data_root>/models/weights.hdf5`。
        train_data_dir: 训练集图片目录（需按类别分子目录，`flow_from_directory` 格式）。
        test_data_dir: 验证集图片目录（同上）。
    """

    def __init__(self, data_root: str, train_data_dir: str, test_data_dir: str) -> None:
        self.checkpoint_path = os.path.join(data_root, "models", "weights.hdf5")
        self.tensorboard_path = os.path.join(
            data_root, "logs", f"kaggle_deepfake-{int(time.time())}"
        )

        self.img_height, self.img_width = 150, 150

        self.training_data_dir = train_data_dir
        self.testing_data_dir = test_data_dir

        self.model: tf.keras.Model | None = None

        self._init()

    def _init(self) -> None:
        """确保权重目录与 tensorboard 日志目录存在。"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)

    def build(self) -> tf.keras.Model:
        """构建三层卷积网络，返回并缓存模型实例。"""
        input_layer = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
        cov1 = Convolution2D(
            32,
            (3, 3),
            name="cov1",
            input_shape=(self.img_width, self.img_height, 3),
            kernel_regularizer=regularizers.l2(0.0005),
            # activity_regularizer=regularizers.l1(0.01),
            activation="relu",
        )(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(cov1)
        cov2 = Convolution2D(
            32,
            (3, 3),
            name="cov2",
            kernel_regularizer=regularizers.l2(0.0005),
            # activity_regularizer=regularizers.l1(0.01),
            activation="relu",
        )(pool1)
        poo2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(cov2)
        cov3 = Convolution2D(
            64,
            (3, 3),
            name="cov3",
            kernel_regularizer=regularizers.l2(0.0005),
            # activity_regularizer=regularizers.l1(0.01),
            activation="relu",
        )(poo2)
        pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(cov3)

        fla = Flatten()(pool3)
        dense1 = Dense(
            64,
            name="dense1",
            kernel_regularizer=regularizers.l2(0.0005),
            # activity_regularizer=regularizers.l1(0.01),
            activation="relu",
        )(fla)
        drop1 = Dropout(0.5)(dense1)
        dense2 = Dense(
            1,
            name="dense2",
            kernel_regularizer=regularizers.l2(0.0005),
            # activity_regularizer=regularizers.l1(0.01),
            activation="sigmoid",
        )(drop1)

        self.model = tf.keras.Model(input_layer, dense2)
        return self.model

    def load(self) -> None:
        """若 `checkpoint_path` 存在已保存权重，则加载。"""
        if os.path.exists(self.checkpoint_path):
            self.model = load_model(self.checkpoint_path)

    def train(self, batch_size: int = 64) -> None:
        """使用 `training_data_dir` / `testing_data_dir` 训练模型并保存权重。

        Args:
            batch_size: 训练与验证的 batch 大小。
        """
        checkpoint = ModelCheckpoint(
            self.checkpoint_path, monitor="val_auc", verbose=1, mode="max"
        )

        tensorboard = TensorBoard(
            log_dir=self.tensorboard_path,
            update_freq=10,
            write_graph=True,
            write_images=True,
            profile_batch=0,
        )

        train_data = ImageDataGenerator(
            rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
        )

        test_data = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_data.flow_from_directory(
            self.training_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode="binary",
        )

        validation_generator = test_data.flow_from_directory(
            self.testing_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode="binary",
        )

        if self.model is None:
            raise RuntimeError("请先调用 build() 或 load() 初始化模型")

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy", keras.metrics.AUC()],
        )

        self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=100,
            callbacks=[tensorboard, checkpoint],
        )

    def clear(self) -> None:
        """删除已保存的权重文件。"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def predict(
        self, predict_dir: str, submission_path: str, batch_size: int = 64
    ) -> pd.DataFrame:
        """对 `predict_dir` 下的图片推理，按视频聚合后写出 Kaggle 提交文件。

        Args:
            predict_dir: 待预测图片目录。
            submission_path: 生成的 `submission.csv` 保存路径，父目录不存在会自动创建。
            batch_size: 推理 batch 大小。

        Returns:
            按 `filename` 聚合后的预测结果 DataFrame，包含 `filename`、`label` 两列。
        """
        test_data = ImageDataGenerator(rescale=1.0 / 255)
        predict_generator = test_data.flow_from_directory(
            predict_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )
        logger.info(f"predict filenames: {predict_generator.filenames}")
        filenames = predict_generator.filenames
        result = []

        for data in tqdm(predict_generator):
            res = self.model.predict(data[0]).transpose()
            temp = np.array([res[0], data[1]]).transpose()

            result.extend(temp)
            if len(result) >= len(filenames):
                result = result[: len(filenames)]
                break

        df0 = list(np.array(result).transpose().tolist())
        df0.append(filenames[: len(result)])
        df1 = np.array(df0).transpose()
        df2 = pd.DataFrame(df1)
        df2.columns = ["label", "real", "name"]
        df2["label"] = df2["label"].astype("float")
        df2["real"] = df2["real"].astype("float")
        df2["filename"] = df2["name"].str.extract(r"[-]([a-z]+)") + ".mp4"
        df2 = df2[["filename", "label", "real"]]

        df3 = df2.groupby("filename").mean().reset_index()
        logger.info(f"submission preview:\n{df3}")

        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        df3[["filename", "label"]].to_csv(submission_path, index=None)
        return df3
