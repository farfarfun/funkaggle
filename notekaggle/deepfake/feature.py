import glob
import os
import time

import cv2
import demjson
import face_recognition
import numpy as np
import pandas as pd
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

    def file_feature(self, result=None):
        """
        另外，cv2.VideoCapture(0)是打开本地摄像头

        以下是opencv-python可以获取视频的相关信息，可以通过从0开始的序号获取
        CV_CAP_PROP_POS_MSEC 视频文件的当前位置（以毫秒为单位）或视频捕获时间戳。
        CV_CAP_PROP_POS_FRAMES 接下来要解码/捕获的帧的基于0的索引。
        CV_CAP_PROP_POS_AVI_RATIO 视频文件的相对位置：0 - 电影的开始，1 - 电影的结尾。
        CV_CAP_PROP_FRAME_WIDTH 视频流中帧的宽度。
        CV_CAP_PROP_FRAME_HEIGHT 视频流中帧的高度。
        CV_CAP_PROP_FPS 帧速率。
        CV_CAP_PROP_FOURCC 编解码器的4字符代码。
        CV_CAP_PROP_FRAME_COUNT 视频文件中的帧数。
        CV_CAP_PROP_FORMAT 返回的Mat对象的格式 retrieve() 。
        CV_CAP_PROP_MODE 指示当前捕获模式的特定于后端的值。
        CV_CAP_PROP_BRIGHTNESS 图像的亮度（仅适用于相机）。
        CV_CAP_PROP_CONTRAST 图像对比度（仅适用于相机）。
        CV_CAP_PROP_SATURATION 图像的饱和度（仅适用于相机）。
        CV_CAP_PROP_HUE 图像的色调（仅适用于相机）。
        CV_CAP_PROP_GAIN 图像的增益（仅适用于相机）。
        CV_CAP_PROP_EXPOSURE 曝光（仅适用于相机）。
        CV_CAP_PROP_CONVERT_RGB 布尔标志，指示是否应将图像转换为RGB。
        CV_CAP_PROP_WHITE_BALANCE_U 白平衡设置的U值（注意：目前仅支持DC1394 v 2.x后端）
        CV_CAP_PROP_WHITE_BALANCE_V 白平衡设置的V值（注意：目前仅支持DC1394 v 2.x后端）
        CV_CAP_PROP_RECTIFICATION 立体摄像机的整流标志（注意：目前仅支持DC1394 v 2.x后端）
        CV_CAP_PROP_ISO_SPEED摄像机 的ISO速度（注意：目前仅支持DC1394 v 2.x后端）
        CV_CAP_PROP_BUFFERSIZE 存储在内部缓冲存储器中的帧数（注意：目前仅支持DC1394 v 2.x后端）
        """

        if result is None:
            result = {}

        file_name = result['video']
        cap = cv2.VideoCapture(self.dir_path + '/' + file_name)
        if cap.isOpened():
            # get方法参数按顺序对应下表（从0开始编号)
            result['frame_rate'] = cap.get(5)  # 帧速率
            result['frame_num'] = cap.get(7)  # 视频文件的帧数
            result['duration'] = result['frame_num'] / result['frame_rate'] / 60  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟

        cap.release()
        return result

    def decode(self, meta, desc=''):
        file_name = meta['video']
        file_path = self.dir_path + '/' + file_name
        v_cap = cv2.VideoCapture(file_path)

        file_name = os.path.basename(file_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        features = []

        for j in tqdm(range(v_len), total=v_len, desc='{}-{}'.format(desc, file_name)):
            v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    frame = cv2.resize(frame, (0, 0), fx=self.resize, fy=self.resize)

                locs = face_recognition.face_locations(frame)
                feas = face_recognition.face_encodings(frame, known_face_locations=locs)

                if len(locs) == 0 or len(feas) == 0:
                    continue
                feature = [j]
                feature.extend(list(locs[0]))
                feature.extend(list(feas[0]))

                features.append(feature)
        v_cap.release()

        if len(features) == 0:
            return None

        res = pd.DataFrame(features)

        cols = ['num']
        cols.extend(['loc_' + str(i + 1) for i in range(0, 4)])
        cols.extend(['encode_' + str(i + 1) for i in range(0, 128)])

        res.columns = cols

        res['video'] = file_name
        return res

    def run(self, overwrite=False):
        d1 = pd.read_csv(self.feature_file_path)
        d2 = d1.to_dict(orient='index')

        result = None
        exist_file = []
        if os.path.exists(self.feature_file_path2):
            result = pd.read_csv(self.feature_file_path2)
            exist_file.extend(result[['video']].drop_duplicates().values.tolist())
            exist_file = [i[0] for i in exist_file]

        i = len(exist_file)

        for key in d2.keys():
            name = d2[key]['video']
            if name in exist_file:
                continue
            i += 1
            temp = self.decode(d2[key],
                               desc='{}/{}-{}'.format(i,
                                                      len(d2.keys()),
                                                      time.strftime("%Y-%m-%d %H:%M:%S",
                                                                    time.localtime())))

            result = pd.concat([result, temp], sort=True)
            result.to_csv(self.feature_file_path2)

    def run2(self, overwrite=False):
        d1 = demjson.decode(open(self.video_json).read())
        d2 = pd.DataFrame.from_dict(d1, orient='index')
        d2.reset_index(inplace=True)
        d2['video'] = d2['index']
        d2['index'] = self.train_index
        d2 = d2[['index', 'split', 'video', 'label', 'original']]
        d1 = d2.to_dict(orient='index')

        if not overwrite and os.path.exists(self.feature_file_path):
            print('file exist return')
            return

        for key in tqdm(d1.keys(), total=len(d1), desc='file_features'):
            value = d1[key]
            d1[key] = self.file_feature(value)

        d2 = pd.DataFrame.from_dict(d1, orient='index')

        d2.to_csv(self.feature_file_path)

    def video2img(self):
        d1 = demjson.decode(open(self.video_json).read())
        d2 = pd.DataFrame.from_dict(d1, orient='index')
        d2.reset_index(inplace=True)

        video_dir = self.dir_path
        save_dir = self.dir_path + '_valid'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for line in tqdm(d2.values):
            video = line[0]
            label = 0 if line[1] == 'FAKE' else 1

            if label == 1:
                self.n_frames = 20
            else:
                self.n_frames = 3

            image_dir = os.path.join(save_dir, str(label))
            video_path = os.path.join(video_dir, video)
            self.video2img_file(video_path, image_dir=image_dir, n_frames=self.n_frames)

    def video2img_file(self, video_path, image_dir, n_frames=10):
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)
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


def run():
    for index in [0, 1]:
        path = '/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_{}'.format(index)
        detection_pipeline = DetectionPipeline(path, detector=None, batch_size=60, resize=0.25, n_frames=30)
        # detection_pipeline.run(overwrite=True)
        detection_pipeline.video2img()


def test_data():
    path = '/Users/liangtaoniu/tmp/dataset/deepfake/dfdc_train_part_1'
    detection_pipeline = DetectionPipeline(path, detector=None, batch_size=60, resize=0.25, n_frames=30)

    path = '/Users/liangtaoniu/tmp/dataset/deepfake/test/deepfake-detection-challenge/test_videos'
    detection_pipeline.test_data(path)


# run()
test_data()
