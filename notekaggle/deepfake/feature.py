import glob
import os

import cv2
import demjson
import numpy as np
import pandas as pd
from face_recognition import face_locations
from tqdm import tqdm


class DetectionPipeline:

    def __init__(self,
                 dir_path,
                 detector,
                 n_frames=None,
                 batch_size=60,
                 resize=0.25):
        self.dir_path = dir_path
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

        self.video_paths = glob.glob(self.dir_path + '/*.mp4')
        self.video_json = self.dir_path + '/metadata.json'

        self.train_index = os.path.basename(self.dir_path).split('_')[-1]
        self.dir_feature = '/Users/liangtaoniu/workspace/MyDiary/src/kaggle/features'

        self.feature_file_path = self.dir_feature + '/file_feature_{}.csv'.format(self.train_index)
        self.feature_file_path2 = self.dir_feature + '/file_feature2_{}.csv'.format(self.train_index)

    def video2img(self):
        d1 = demjson.decode(open(self.video_json).read())
        d2 = pd.DataFrame.from_dict(d1, orient='index')
        d2.reset_index(inplace=True)

        video_dir = self.dir_path
        save_dir = self.dir_path + '_img'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        index = 1000000
        for line in tqdm(d2.values):
            if line[1] == 'REAL':
                continue
            index += 1
            _fake = line[0]
            _real = line[3]

            real_dir = os.path.join(save_dir, '1')
            real_path = os.path.join(video_dir, _real)
            self.video2img_file2(real_path, image_dir=real_dir, n_frames=5, index=index)

            fake_dir = os.path.join(save_dir, '0')
            fake_path = os.path.join(video_dir, _fake)
            self.video2img_file2(fake_path, image_dir=fake_dir, n_frames=5, index=index)

    def video2img_file2(self, video_path, image_dir, n_frames=10, index=1000000):
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
                # print((top, right, bottom, left))

                frame = frame[top:bottom, left:right]

                cv2.imwrite(image_dir + '/{}-{}-{}.jpg'.format(index, 1000 + i, video_name.split('.')[0]), frame)
        cap.release()
        # break

    def video2img_file(self, video_path, image_dir, n_frames=10):
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample = np.linspace(0, v_len - 1, n_frames).astype(int)
        for i in range(v_len):
            ret, frame = cap.read()
            if ret and i in sample:
                cv2.imwrite(image_dir + '/{}-{}.jpg'.format(video_name.split('.')[0], 10000 + i), frame)
        cap.release()
        # break

    def test_data(self, path):
        paths = os.listdir(path)
        print(len(paths))
        for name in os.listdir(path):
            video_path = '{}/{}'.format(path, name)
            print(video_path)
            self.video2img_file(video_path=video_path, image_dir=path + '_img', n_frames=10)


def run(data_root=None, index=0):
    data_root = data_root or '/Users/liangtaoniu/tmp/dataset/deepfake/'

    path = data_root + '/dfdc_train_part_{}'.format(index)
    detection_pipeline = DetectionPipeline(path, detector=None, batch_size=60, resize=0.25, n_frames=30)

    detection_pipeline.video2img()


def data_test():
    path = '/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_0'
    detection_pipeline = DetectionPipeline(path, detector=None, batch_size=60, resize=0.25, n_frames=30)

    path = '/Users/liangtaoniu/tmp/dataset/deepfake/test/deepfake-detection-challenge/test_videos'
    detection_pipeline.test_data(path)

# run()
# data_test()
