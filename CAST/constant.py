#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: constant.py
@time: 10/4/19 6:22 PM
@version 1.0
@desc:
"""
import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_MODE = 'Train'
VAL_MODE = 'Validate'
TEST_MODE = 'Test'

colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                     [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                     [0, 192, 0], [128, 192, 0], [0, 64, 128]])

FACE_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA')
FACE_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/img')
FACE_MASK_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/mask')
FACE_DATASET_COLOR_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/mask-color')
FACE_WARPED = os.path.join(PROJECT_DIR, 'datasets/CelebA/warped')
FACE_LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/landmarks')
FACE_LANDMARKS_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/img-landmarks')
FACE_WCT_LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/wct-landmarks')
FACE_WCT_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/wct')
FACE_WCT_CMP_PATH = os.path.join(PROJECT_DIR, 'datasets/CelebA/cmp')
FACE_WCT_PSM_PATH = 'datasets/CelebA/wct-psm'
# FACE_WCT_PSM_PATH = '/data/yiyuiii/wct-psm'
FACE_WCT_MASK = os.path.join(PROJECT_DIR, 'datasets/CelebA/wct-mask')
FACE_WCT_MASK_COLOR = os.path.join(PROJECT_DIR, 'datasets/CelebA/wct-color')

CARI_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari')
CARI_MASK_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari/mask')
CARI_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari/img')
CARI_DATASET_COLOR_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari/mask-color')
CARI_LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari/landmarks')
CARI_LANDMARKS_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/Cari/img-landmarks')

A1_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger')
A1_MASK_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger/mask')
A1_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger/img')
A1_DATASET_COLOR_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger/mask-color')
A1_LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger/landmarks')
A1_LANDMARKS_IMG_PATH = os.path.join(PROJECT_DIR, 'datasets/Fernand_Leger/img-landmarks')

WARP_PSM_PATH = os.path.join(PROJECT_DIR, 'datasets/Warp-psm')

IMG_SIZE = 512
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/config-server.yaml')
TEST_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/config-test.yaml')

SERVER_WARPED_PSM_PATH = '/data/lxd/Warp-psm'
# SERVER_WARPED_PSM_PATH = '/data/yiyuiii/Warp-psm'
FACE_SAMPLE_OFFSET = 500

# sample_num_list = [50, 50, 30, 30, 30, 30, 30, 30, 30, 30]
sample_num_list = [80, 50, 50, 25, 25, 25, 25, 30, 20, 20]


class DatasetPath:

    def __init__(self, root, img, mask, color, landmarks, landmarks_img, warped=None, wct_img=None, wct_landmarks=None,
                 wct_cmp=None, wct_psm=None, wct_mask=None, wct_color=None, with_keypoints=None):
        self.root = root
        self.img = img
        self.mask = mask
        self.color = color
        self.landmarks = landmarks
        self.landmarks_img = landmarks_img
        self.warped = warped
        self.wct_img = wct_img
        self.wct_landmarks = wct_landmarks
        self.wct_cmp = wct_cmp
        self.wct_psm = wct_psm
        self.wct_mask = wct_mask
        self.wct_color = wct_color


celeb_path = DatasetPath(FACE_PATH, FACE_IMG_PATH, FACE_MASK_PATH, FACE_DATASET_COLOR_PATH,
                         FACE_LANDMARKS_PATH, FACE_LANDMARKS_IMG_PATH, FACE_WARPED, FACE_WCT_IMG_PATH,
                         FACE_WCT_LANDMARKS_PATH,
                         FACE_WCT_CMP_PATH, FACE_WCT_PSM_PATH, FACE_WCT_MASK, FACE_WCT_MASK_COLOR)

cari_path = DatasetPath(CARI_PATH, CARI_IMG_PATH, CARI_MASK_PATH, CARI_DATASET_COLOR_PATH, CARI_LANDMARKS_PATH,
                        CARI_LANDMARKS_IMG_PATH)

a1_path = DatasetPath(A1_PATH, A1_IMG_PATH, A1_MASK_PATH, A1_DATASET_COLOR_PATH, A1_LANDMARKS_PATH,
                      A1_LANDMARKS_IMG_PATH,)
