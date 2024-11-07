#!/usr/bin/env python
import argparse
import cv2
import pandas as pd
import numpy as np
import h5py
import json
import csv
import os

# raw_train_path = './comma_dataset/camera/2016-06-08--11-46-01.h5'
# log_train_path = './comma_dataset/log_csv/2016-06-08--11-46-01.csv'

# cam = h5py.File(raw_train_path, 'r')
# log = pd.read_csv(log_train_path)

img = cv2.imread('/home/dungrup/dhruva-vol2/SuperRes/comma_cnn_enhance/2016-06-08--11-46-01/0001310.png')
cv2.imshow('jpg Image', img)
cv2.waitKey(0)


cv2.destroyAllWindows()
# cam.close()