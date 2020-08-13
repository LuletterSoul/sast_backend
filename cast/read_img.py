#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: read_img.py
@time: 5/18/20 5:47 PM
@version 1.0
@desc:
"""
import cv2
import numpy as np
import os
from pathlib import Path

input_dir = 'output/articts'
art_name = 'Hindu_Gods'
txt_dir = 'data/a2.txt'
output_dir = 'output/contents_a2'

Path(output_dir).mkdir(exist_ok=True, parents=True)

with open(txt_dir, 'r') as f:
    pairs = [t.split('\t') for t in f.readlines()]
    for p in pairs:
        img_path = os.path.join(input_dir, art_name, 'warp', p[0], p[1].replace('\n', '') + '.png')
        img = cv2.imread(img_path)
        if img is None:
            print(f'Empty Image: {img_path}')
            continue
        cv2.imwrite(os.path.join(output_dir, art_name + '-' + p[0] + '_' + p[1].replace('\n', '') + '.png'), img)
