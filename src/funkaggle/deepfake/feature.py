"""从 DFDC 视频中抽取人脸帧，按真假标签分目录保存为图片。"""

from __future__ import annotations

import os

import cv2
import demjson3
import numpy as np
import pandas as pd
from face_recognition import face_locations
from tqdm import tqdm


def video2img_file(
    video_path: str, image_dir: str, n_frames: int = 5, index: int = 1000000
) -> None:
    """从单个视频里均匀抽取 `n_frames` 帧，裁出人脸区域后保存为 jpg。

    Args:
        video_path: 视频文件路径。
        image_dir: 输出图片目录，不存在则自动创建。
        n_frames: 抽取帧数。
        index: 输出文件名前缀，用于避免多视频间文件名冲突。
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    for i in range(v_len):
        ret, frame = cap.read()
        if ret and i in sample:
            loc = face_locations(frame)
            if len(loc) == 0:
                continue
            (top, right, bottom, left) = loc[0]
            frame = frame[top:bottom, left:right]

            cv2.imwrite(
                image_dir
                + "/{}-{}-{}.jpg".format(index, 1000 + i, video_name.split(".")[0]),
                frame,
            )
    cap.release()


def video2img_train(
    source_dir: str, target_dir: str | None = None, n_frames: int = 5
) -> None:
    """按 `source_dir/metadata.json` 里的真假标签，为训练集批量抽帧。

    真实视频（`1/` 子目录）与对应的伪造视频（`0/` 子目录）用相同 index 关联。

    Args:
        source_dir: 视频与 `metadata.json` 所在目录。
        target_dir: 输出根目录，默认 `<source_dir>_img`。
        n_frames: 每个视频抽取帧数。
    """
    video_json = source_dir + "/metadata.json"
    target_dir = target_dir or source_dir + "_img"

    with open(video_json, encoding="utf-8") as f:
        d1 = demjson3.decode(f.read())
    d2 = pd.DataFrame.from_dict(d1, orient="index")
    d2.reset_index(inplace=True)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    index = 1000000
    for line in tqdm(d2.values):
        if line[1] == "REAL":
            continue
        index += 1
        _fake = line[0]
        _real = line[3]

        real_dir = os.path.join(target_dir, "1")
        real_path = os.path.join(source_dir, _real)
        video2img_file(real_path, image_dir=real_dir, n_frames=n_frames, index=index)

        fake_dir = os.path.join(target_dir, "0")
        fake_path = os.path.join(source_dir, _fake)
        video2img_file(fake_path, image_dir=fake_dir, n_frames=n_frames, index=index)


def video2img_predict(
    source_dir: str, target_dir: str | None = None, n_frames: int = 5
) -> None:
    """为待预测视频批量抽帧。

    Args:
        source_dir: 视频所在目录。
        target_dir: 输出根目录，默认 `<source_dir>_img`。
        n_frames: 每个视频抽取帧数。
    """
    target_dir = target_dir or source_dir + "_img"

    for index, name in enumerate(tqdm(os.listdir(source_dir)), start=1):
        video_path = f"{source_dir}/{name}"

        real_dir = os.path.join(target_dir, "1")
        video2img_file(video_path, image_dir=real_dir, n_frames=n_frames, index=index)
