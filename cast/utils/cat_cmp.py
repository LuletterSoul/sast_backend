#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: generate.py
@time: 4/26/20 2:35 PM
@version 1.0
@desc:
"""

import numpy as np
import cv2
from pathlib import Path
import os

input_dir = '/Users/luvletteru/Documents/cmp'

output_dir = Path('/Users/luvletteru/Documents/cmp')

output_dir.mkdir(parents=True, exist_ok=True)

image_paths = os.listdir(input_dir)

idx = 0
plot = []
h_plot = []
row_internal = 4
row = 0
col = 4
cnt = 0
for idx, ip in enumerate(image_paths):
    img = cv2.imread(os.path.join(input_dir, ip))
    print(idx)
    print(os.path.join(input_dir,ip))
    if img is not None:
        h_plot.append(img)
        if (idx + 1) % col == 0:
            h_plot = np.hstack(h_plot)
            plot.append(h_plot)
            h_plot = []
            row += 1
        if (row + 1) % row_internal == 0:
            plot = np.vstack(plot)
            cv2.imwrite(os.path.join(output_dir, str(cnt) + '.png'), plot)
            plot = []
            row = 0
            cnt += 1
if len(plot):
    plot = np.vstack(plot)
    cv2.imwrite(os.path.join(output_dir, str(cnt) + '.png'), plot)
