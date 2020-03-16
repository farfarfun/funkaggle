from setuptools import setup, find_packages

install_requires = ['kaggle', 'notetool', 'demjson', 'cv2']

setup(name='notekaggle',
      version='0.0.1',
      description='notekaggle',
      author='niuliangtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )
