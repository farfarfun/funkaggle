import os

import cv2
import demjson
import numpy as np
import pandas as pd
from face_recognition import face_locations
from tqdm import tqdm


def video2img_file(video_path, image_dir, n_frames=5, index=1000000):
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
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

            cv2.imwrite(image_dir + '/{}-{}-{}.jpg'.format(index, 1000 + i, video_name.split('.')[0]), frame)
    cap.release()


def video2img_train(source_dir, target_dir=None, n_frames=5):
    video_json = source_dir + '/metadata.json'
    target_dir = target_dir or source_dir + '_img'

    d1 = demjson.decode(open(video_json).read())
    d2 = pd.DataFrame.from_dict(d1, orient='index')
    d2.reset_index(inplace=True)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    index = 1000000
    for line in tqdm(d2.values):
        if line[1] == 'REAL':
            continue
        index += 1
        _fake = line[0]
        _real = line[3]

        real_dir = os.path.join(target_dir, '1')
        real_path = os.path.join(source_dir, _real)
        video2img_file(real_path, image_dir=real_dir, n_frames=n_frames, index=index)

        fake_dir = os.path.join(target_dir, '0')
        fake_path = os.path.join(source_dir, _fake)
        video2img_file(fake_path, image_dir=fake_dir, n_frames=n_frames, index=index)


def video2img_predict(source_dir, target_dir=None, n_frames=5):
    target_dir = target_dir or source_dir + "_img"

    index = 0
    for name in os.listdir(source_dir):
        index += 1
        video_path = '{}/{}'.format(source_dir, name)

        real_dir = os.path.join(target_dir, '1')
        video2img_file(video_path, image_dir=real_dir, n_frames=n_frames, index=index)
