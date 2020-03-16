import os

from notetool.download import download


def get_url(file_index=0):
    url_base = 'https://storage.googleapis.com/kaggle-competitions-detached-data/16880'
    url_file = 'dfdc_train_part_{}.zip'.format(str(file_index).rjust(2, '0'))
    url_para = 'GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584504923&Signature=PZhFd0EWeG27T6Km3esWUsjASDUjw2uPiI7W%2BforX5krTX1T9iXuTDIr5hS6y87Fsdky0Y07RmHBSreWL0lf7jkOdxJwPxDU%2FCpmaHph00ZQSGg2SUcQw8IQXtqj%2FQ3t%2B4wRvsc7JvkhkfFy24PH8wQt2udqEh6MOcTFUL6MHofC%2BlZzhmQza6sgyWqDJLQlmPCMCvq%2ByzeYB5t4jvYknAO5w5%2FQvegfCnElcV0wCaufgfgJQjERgJ5RiTELOPh7dZZgMA7fAOHFrKoWX%2F7iXKVJ%2Fs9rXGmpFdv%2F02IGxQbEozGC%2F%2FQaT40zumAp3YmGnUgaktHJ2q6E5YlF6hOl%2BQ%3D%3D'
    _url = '{}/{}?{}'.format(url_base, url_file, url_para)
    return _url


def download_file(file_index=0, save_dir='/root/dataset/deepfake'):
    file_name = 'dfdc_train_part_{}.zip'.format(str(file_index).rjust(2, '0'))
    save_path = os.path.join(save_dir, file_name)
    url = get_url(file_index)
    download(url, save_path)
    os.system('unzip -n {}'.format(save_path))
