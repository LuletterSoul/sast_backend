#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: fmt.py
@time: 2/22/20 10:26 AM
@version 1.0
@desc:
"""

import cv2

import os

fmt_dir = 'experiments/02-17-lw1e5_iter500_200_512_ul50_uh50_kl7_km1_maskguided2'
img_paths = os.listdir(fmt_dir)

out_dir = 'fmt-result'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for img_path in img_paths:
    # _, content_name, style_name = os.path.splitext(img_path)[0].split('-')
    res = os.path.splitext(img_path)[0].split('-')
    if len(res) == 3:
        _, content_name, style_name = res
        img = cv2.imread(os.path.join(fmt_dir, img_path))
        cv2.imwrite(os.path.join(out_dir, f'{content_name}-{style_name}.png'), img)
