import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='rl_teacher',
    author='Dagasfi',
    version='0.1.0',
    install_requires=[
        'mujoco-py ~=0.5.7',
        'tqdm',
        'matplotlib',
        'ipython',
        'ipdb',
        'numpy==1.13.3',
        'WhiteNoise==3.2.2',
        'dj-database-url==0.4.0',
        'django==1.11',
        'scipy==1.2.2',
        'protobuf==3.6.1',
        'keras==2.1.6',
        'wrapt',
        'gym==0.10.5',
        'tensorboard==1.10',
        'tensorflow==1.10',
        'Pillow',
        'Flask',
    ],
    # https://github.com/tensorflow/tensorflow/issues/7166#issuecomment-280881808
    extras_require={
        "tf": ["tensorflow ~= 1.10"],
        "tf_gpu": ["tensorflow-gpu >= 1.1"],
    }
)
