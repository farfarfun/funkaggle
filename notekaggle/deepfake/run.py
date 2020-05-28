from notekaggle.deepfake.download import download_files
from notekaggle.deepfake.feature import video2img_predict
from notekaggle.deepfake.feature import video2img_train
from notekaggle.deepfake.model import MyModel

# data_root = '/root/dataset/deepfake'
data_root = '/Users/liangtaoniu/tmp/dataset/deepfake/'
train_data = data_root + 'train_data'
test_data = data_root + 'test_data'
predict_data = data_root + 'predict_data'

predict_path = data_root + '/deepfake-detection-challenge/test_videos'


def download():
    l = list(range(5))
    download_files(save_dir=data_root, file_index_list=l)


def feature(test=False, train=False, predict=False):
    if test:
        for index in [0]:
            path = data_root + '/dfdc_train_part_{}'.format(index)
            video2img_train(path, target_dir=test_data)
    if train:
        for index in [2, 3]:
            path = data_root + '/dfdc_train_part_{}'.format(index)
            video2img_train(path, target_dir=train_data)
    if predict:
        video2img_predict(predict_path, target_dir=predict_data)


def model_train():
    model = MyModel(data_root=data_root,
                    train_data_dir=train_data,
                    test_data_dir=test_data
                    )
    model.build()
    # model.clear()
    model.load()
    model.train(batch_size=32)


def model_predict():
    model = MyModel(data_root=data_root,
                    train_data_dir=train_data,
                    test_data_dir=test_data
                    )
    model.build()

    model.load()
    model.predict(predict_dir=predict_data)


# model_train()
# feature(predict=True)
model_predict()
