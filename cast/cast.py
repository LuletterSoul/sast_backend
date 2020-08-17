#!/usr/bin/env python
# encoding: utf-8
"""
@author: Kris Hao 孔美豪
@license: (C) Copyright 2020-now, Node Supply Chain Manager Corporation Limited.
@contact: 1178143318@qq.com
@software:
@file: cast.py
@time: 8/9/20 6:06 PM
@version 1.0
@desc:
"""
import os
import time
import uuid
from pathlib import Path
from random import sample

import cv2
import numpy as np
import torch

from config import Config
from .utils import whiten_and_color, warp_image
from utils import *


def warp_interplation_from_datasets(web_cari_path: Path, train_photo_num, train_cari_num,
                                    test_photo_num, test_cari_num, output_path,
                                    fmt='.jpg', web_test_photo_path=None, web_test_cari_path=None):
    """
    形变线性插值实验
    :param web_cari_path:
    :param train_photo_num:
    :param train_cari_num:
    :param test_photo_num:
    :param test_cari_num:
    :param output_path:
    :param fmt:
    :param web_test_photo_path:
    :param web_test_cari_path:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'
    # web_cari_path = Path('datasets/WebCari_512')
    # warp_interplation(web_cari_path, 10, 10, 100, 100, f'output/random/{generate_time_stamp()}')
    # 上面是调用这个函数时，传入的参数
    Path(output_path).mkdir(exist_ok=True, parents=True)
    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(128)
    photo_paths = list(web_cari_photo_path.glob(f'*/P*{fmt}'))
    # 列出P*目录下所有的jpg格式文件，并且返回参数为一个列表list
    cari_paths = list(web_cari_photo_path.glob(f'*/C*{fmt}'))
    # shuffles the datasets
    np.random.shuffle(photo_paths)
    np.random.shuffle(cari_paths)
    train_photo_paths = sample(photo_paths, train_photo_num)
    # 从上面返回的list中随机抽取train_photo_num（10）个jpg图片
    train_cari_paths = sample(cari_paths, train_cari_num)

    if web_test_photo_path is not None:
        # test_photo_paths = list(Path(web_test_photo_path).glob('P*'))
        test_photo_paths = [web_test_photo_path]
    else:
        test_photo_paths = sample(photo_paths, test_photo_num)

    if web_test_cari_path is not None:
        # test_cari_paths = list(Path(web_test_cari_path).glob('*'))
        test_cari_paths = [web_test_cari_path]
    else:
        test_cari_paths = sample(cari_paths, test_cari_num)
        # 默认的情况就是test的图片采样自photo图片

    print(f'Photo num {len(train_photo_paths)}')
    print(f'Cari num {len(train_cari_paths)}')

    train_face_landmarks, train_face_paths = load_landmarks(train_photo_paths, web_cari_landmarks_path)
    # 返回的第一个值为每一个photo（C开头的jpg）对应的landmark构成的list，第二个值就是train_photo_paths
    train_cari_landmarks, train_cari_paths = load_landmarks(train_cari_paths, web_cari_landmarks_path)
    # 返回的第一个值为每一个漫画（P开头的jpg）对应的landmark构成的list，第二个值就是train_photo_paths
    train_photo_num = len(train_face_paths)
    train_cari_num = len(train_cari_paths)

    if train_photo_num == 0 or train_cari_num == 0:
        raise Exception('Could not find any landmarks')

    for idx, photo_path in enumerate(test_photo_paths):
        if idx > len(test_photo_paths) - 1:
            break
        test_face_landmark, test_face_path = load_landmarks([test_photo_paths[idx]], web_cari_landmarks_path)
        test_cari_landmark, test_cari_path = load_landmarks([test_cari_paths[idx]], web_cari_landmarks_path)
        if not len(test_face_path) or not len(test_cari_path):
            continue

        test_photo_tensor = cvt_landmarks_distribution(device, 1, test_face_landmark)
        test_cari_tensor = cvt_landmarks_distribution(device, 1, test_cari_landmark)

        print(test_photo_tensor.size())
        print(test_cari_tensor.size())
        wct_landmark_1, test_wct_landmark_1 = whiten_and_color(test_photo_tensor, test_cari_tensor,
                                                               test_photo_tensor, use_mean=True)
        face = cv2.imread(str(photo_path))
        cari_1 = cv2.imread(str(test_cari_path[0]))

        test_face_landmark = test_face_landmark.reshape((1, 128, 2))
        test_wct_landmark_1 = test_wct_landmark_1.permute(1, 0).view(1, 128, 2).long().cpu().numpy()

        warped_1, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_1[0])
        warped_1 = (warped_1 * 255).astype(np.uint8)
        # output = [face, cari_1, warped_1]
        output = [warped_1]
        output = np.hstack(output)
        cv2.imwrite(f'{output_path}/content.jpg', output)
        cv2.imwrite('images/style/style.jpg', cari_1)


def warp_interplation_from_images(content_id, style_id, images_path: Path, train_photo_num, train_cari_num,
                                  test_photo_num, test_cari_num, output_path,
                                  fmt='.jpg', web_test_photo_path=None, web_test_cari_path=None):
    # images_path = images
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Path(output_path).mkdir(exist_ok=True, parents=True)

    test_photo_paths = [os.path.join(Config.CONTENT_DIRECTORY, content_id)]
    test_cari_paths = [os.path.join(Config.STYLE_DIRECTORY, style_id)]
    landmarks_path = Path(Config.LANDMARK_DIRECTORY)
    output_path = Config.CONTENT_DIRECTORY
    for idx, photo_path in enumerate(test_photo_paths):
        if idx > len(test_photo_paths) - 1:
            break
        test_face_landmark, test_face_path = load_landmarks_from_images([test_photo_paths[idx]], landmarks_path)
        test_cari_landmark, test_cari_path = load_landmarks_from_images([test_cari_paths[idx]], landmarks_path)
        if not len(test_face_path) or not len(test_cari_path):
            raise Exception(f'Could not find any landmarks from {test_photo_paths[idx]} '
                            f'or {test_cari_paths[idx]}')

        test_photo_tensor = cvt_landmarks_distribution(device, 1, test_face_landmark)
        test_cari_tensor = cvt_landmarks_distribution(device, 1, test_cari_landmark)

        print(test_photo_tensor.size())
        print(test_cari_tensor.size())
        wct_landmark_1, test_wct_landmark_1 = whiten_and_color(test_photo_tensor, test_cari_tensor,
                                                               test_photo_tensor, use_mean=True)
        face = cv2.imread(str(photo_path))
        cari_1 = cv2.imread(str(test_cari_path[0]))

        test_face_landmark = test_face_landmark.reshape((1, 128, 2))
        test_wct_landmark_1 = test_wct_landmark_1.permute(1, 0).view(1, 128, 2).long().cpu().numpy()

        warped_1, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_1[0])
        warped_1 = (warped_1 * 255).astype(np.uint8)
        # output = [face, cari_1, warped_1]
        output = [warped_1]
        output = np.hstack(output)
        warped_content_id = f'{get_prefix(content_id)}-{get_prefix(style_id)}_{str(uuid.uuid1())}.png'
        cv2.imwrite(f'{output_path}/{warped_content_id}', output)
        # cv2.imwrite('images/style/style.jpg', cari_1)
        return warped_content_id


def cvt_landmarks_distribution(device, dim, landmarks):
    """
    convert N * landmarks_num * 2 to (landmarks_num * 2) * N
    :param device:
    :param dim:
    :param landmarks:
    :param scale:
    :return:
    """
    distribution = torch.from_numpy(landmarks).view(dim, -1).permute(1, 0).double().to(device)
    # 即将原来dim个图片的landmarks展开成论文中提到的dxnp（dxnc）大小的tensor
    return distribution


def load_landmarks(paths, landmarks_path):
    # paths = datasets / WebCari_512 / img / * / P * {.jpg}
    # landmarks_path = datasets / WebCari_512 / landmarks / 128
    landmarks = []
    filter_paths = []
    for fp in paths:
        print('Loading landmarks [{}]...'.format(fp))
        # {}代表format中的值，{0}代表第一个
        # {}代表format中的值，{0}代表第一个
        # print(' {}	{}aa{}'.format(1, 2, 3))
        # 1   2aa3
        # print(' {2}	{2}aa{2}'.format(1, 2, 3))
        #  3  3aa3
        face_name = os.path.basename(str(fp))
        # os.path.basename返回path最后的文件名,即如P00001.jpg
        index = os.path.splitext(face_name)[0]
        # os.path.splitext分离文件名与扩展名,最后[0]即返回文件名，而丢掉扩展名.jpg，如
        pdir = os.path.basename(os.path.dirname(str(fp)))
        # os.path.dirname去掉最后文件名，返回目录
        landmark_path = landmarks_path / pdir / 'landmarks' / (str(index) + '.txt')
        # 上面即得到每一个图片对应的landmark坐标txt文件
        if landmark_path.exists():
            landmarks.append(
                np.loadtxt(str(landmark_path)))
            # np.loadtxt的功能是读入数据文件，这里的数据文件要求每一行数据的格式相同
            filter_paths.append(fp)
    return np.array(landmarks), filter_paths


def load_landmarks_from_images(paths, landmarks_path):
    # path是挑选出来的十个photo图片，landmarks_path
    # paths = [images/style/style.jpg]
    # landmarks_path = images / landmarks
    landmarks = []
    filter_paths = []
    for fp in paths:
        print('Loading images landmarks [{}]...'.format(fp))
        face_name = os.path.basename(str(fp))
        # os.path.basename返回path最后的文件名,即如P00001.jpg
        index = os.path.splitext(face_name)[0]
        # os.path.splitext分离文件名与扩展名,最后[0]即返回文件名，而丢掉扩展名.jpg，如
        landmark_path = landmarks_path / (str(index) + '.txt')
        # 上面即得到每一个图片对应的landmark坐标txt文件
        if landmark_path.exists():
            landmarks.append(
                np.loadtxt(str(landmark_path)))
            # np.loadtxt的功能是读入数据文件，这里的数据文件要求每一行数据的格式相同
            filter_paths.append(fp)
    return np.array(landmarks), filter_paths


def generate_time_stamp(fmt='%m%d%H%M'):
    return time.strftime(fmt, time.localtime(time.time()))


def warp_content_to_style_datasets(photo_content='datasets/WebCari_512/img/Adele Laurie Blue Adkins/P00004.jpg'
                                   , caricature_style='datasets/WebCari_512/img/Adele Laurie Blue Adkins/C00003.jpg'):
    web_cari_path = Path('datasets/WebCari_512')
    # warp_interplation(web_cari_path, 10, 10, 1, 1, f'output/{generate_time_stamp()}')
    warp_interplation_from_datasets(web_cari_path, 10, 10, 1, 1, f'images/content',
                                    web_test_photo_path=photo_content,
                                    web_test_cari_path=caricature_style)
    # 上面最后两个参数分别为content（photo）图和style（caricature）图


def warp_content_to_style_images(content_id, style_id, photo_content='images/content/content.jpg'
                                 , caricature_style='images/style/style.jpg'):
    images_path = Path('images')
    # warp_interplation(web_cari_path, 10, 10, 1, 1, f'output/{generate_time_stamp()}')
    return warp_interplation_from_images(content_id, style_id, images_path, 10, 10, 1, 1, f'images/content',
                                         web_test_photo_path=photo_content,
                                         web_test_cari_path=caricature_style)
    # 上面最后两个参数分别为content（photo）图和style（caricature）图
