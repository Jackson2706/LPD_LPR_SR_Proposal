# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import archs
import dataset
import models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print("root_path",root_path)
    train_pipeline(root_path)
