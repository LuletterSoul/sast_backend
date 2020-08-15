#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: sort.py
@time: 2019/12/21 13:41
@version 1.0
@desc:
"""
import os
import json
import cv2
from pathlib import Path
import numpy as np


def sort_landmarks_sparse_1(output_dir: Path):
    for d in output_dir.iterdir():
        landmarks_dir = d / 'landmarks'
        json_dir = d / 'results'
        results = list(json_dir.glob('*.json'))
        for r in results:
            print(r)
            res = json.load(open(r))
            if 'faces' not in res:
                return None
            if 'landmark' not in res['faces'][0]:
                return None
            landmarks = res['faces'][0]['landmark']
            landmarks_list = []
            landmarks = sort_dict(landmarks)
            # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
            for k, landmark in landmarks.items():
                landmarks_list.append([landmark['x'], landmark['y']])
            landmarks_list = np.array(landmarks_list)
            img_name = os.path.splitext(os.path.basename(r))[0]
            txt_name = img_name + '.txt'
            np.savetxt(str(landmarks_dir / txt_name), landmarks_list, fmt="%d")


def sort_landmarks_sparse_2(output_dir: Path):
    landmarks_dir = output_dir / 'landmarks'
    json_dir = output_dir / 'results'
    results = list(json_dir.glob('*.json'))
    for r in results:
        print(r)
        res = json.load(open(r))
        if 'faces' not in res:
            return None
        if 'landmark' not in res['faces'][0]:
            return None
        landmarks = res['faces'][0]['landmark']
        landmarks_list = []
        landmarks = sort_dict(landmarks)
        # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
        for k, landmark in landmarks.items():
            landmarks_list.append([landmark['x'], landmark['y']])
        landmarks_list = np.array(landmarks_list)
        img_name = os.path.splitext(os.path.basename(r))[0]
        txt_name = img_name + '.txt'
        np.savetxt(str(landmarks_dir / txt_name), landmarks_list, fmt="%d")


def sort_landmarks_dense_1(output_dir: Path):
    for d in output_dir.iterdir():
        landmarks_dir = d / 'landmarks'
        json_dir = d / 'results'
        results = list(json_dir.glob('*.json'))
        for r in results:
            print(r)
            res = json.load(open(r))
            if 'face' not in res:
                return None
            if 'landmark' not in res['face']:
                return None
            landmarks = res['face']['landmark']
            landmarks_list = []
            # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
            for region, landmarks_dict in landmarks.items():
                landmarks_dict = sort_dict(landmarks_dict)
                for k, landmark in landmarks_dict.items():
                    landmarks_list.append([landmark['x'], landmark['y']])
            landmarks_list = np.array(landmarks_list)
            img_name = os.path.splitext(os.path.basename(r))[0]
            txt_name = img_name + '.txt'
            np.savetxt(str(landmarks_dir / txt_name), landmarks_list, fmt="%d")


def sort_landmarks_dense_2(output_dir: Path):
    landmarks_dir = output_dir / 'landmarks'
    json_dir = output_dir / 'results'
    results = list(json_dir.glob('*.json'))
    for r in results:
        print(r)
        res = json.load(open(r))
        if 'face' not in res:
            return None
        if 'landmark' not in res['face']:
            return None
        landmarks = res['face']['landmark']
        # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
        landmarks_list = []
        for region, landmarks_dict in landmarks.items():
            landmarks_dict = sort_dict(landmarks_dict)
            for k, landmark in landmarks_dict.items():
                landmarks_list.append([landmark['x'], landmark['y']])
        landmarks_list = np.array(landmarks_list)
        img_name = os.path.splitext(os.path.basename(r))[0]
        txt_name = img_name + '.txt'
        np.savetxt(str(landmarks_dir / txt_name), landmarks_list, fmt="%d")


def sort_dict(landmarks_dict):
    landmarks_list = sorted(landmarks_dict.items(), key=lambda d: d[0])
    new_dict = {}
    for entry in landmarks_list:
        new_dict[entry[0]] = entry[1]
    return new_dict


def sortedDictValues(adict):
    keys = adict.keys()
    keys.sort()
    return map(adict.get, keys)


# landmarks_path = 'datasets/Articst-faces/landmarks'
# landmarks_path = 'datasets/WebCariTrain/landmarks/845'
landmarks_path = 'datasets/Articst-faces/landmarks'
# dataset_name = 'AF_dataset'
# output_name = 'AF-landmarks-83'
landmarks_path = Path(landmarks_path)
# sort_landmarks_dense_2(landmarks_path)
sort_landmarks_dense_1(landmarks_path)
# sort_landmarks_sparse_1(landmarks_path)
