#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: group.py
@time: 4/26/20 9:50 AM
@version 1.0
@desc:
"""
from pathlib import Path
import cv2
import numpy as np
import os
import shutil

input_dir = 'datasets/WebCari_512/landmarks/128'
output_dir = 'datasets/WebCari_512_landmarks'
p_output = Path(output_dir) / 'photos'
c_output = Path(output_dir) / 'caris'
p_output.mkdir(parents=True, exist_ok=True)
c_output.mkdir(parents=True, exist_ok=True)

names = os.listdir(input_dir)

for n in names:
    person_path = Path(os.path.join(input_dir, n))
    photos = person_path.glob('*/P*.txt')
    caris = person_path.glob('*/C*.txt')
    new_n = n.replace(' ', '_')
    for p in photos:
        shutil.copy(str(p), str(p_output / f'{new_n}_{os.path.basename(p)}'))
    for c in caris:
        shutil.copy(str(c), str(c_output / f'{new_n}_{os.path.basename(c)}'))
