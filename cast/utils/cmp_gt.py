#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: cmp.py
@time: 4/23/20 9:40 AM
@version 1.0
@desc:
"""
import cv2
import os
import numpy as np
from pathlib import Path

style_dir = 'images/styles_gerwomen_0519'
content_dir = 'images/contents_0421_k'

stylization_dir = 'exp/0520_cw1_lw30_k50_ups50_women'

output_path = Path('output/0520_ger_cmp')

if not output_path.exists():
    output_path.mkdir(exist_ok=True, parents=True)

content_names = os.listdir(content_dir)
style_names = os.listdir(style_dir)

plot = []
first = True
for c in content_names:
    if first:
        h_plot = [np.ones((512, 512, 3), dtype=np.uint8) * 255]
        s1 = style_names[0].split('-')
        for idx, s in enumerate(style_names):
            sp = os.path.join(style_dir, f'{s1[0]}-{s1[1]}-{idx}.png')
            h_plot.append(cv2.imread(sp))
        h_plot = np.hstack(h_plot)
        plot.append(h_plot)
        first = False
    h_plot = [cv2.imread(os.path.join(content_dir, c))]
    stylizations = os.listdir(f'{stylization_dir}/{c[:-4]}')
    for idx, s in enumerate(style_names):
        style_path = f'{stylization_dir}/{c[:-4]}/{c[:-4]}-{s1[0]}-{s1[1]}-{idx}.png'
        print(style_path)
        h_plot.append(cv2.imread(style_path))
    plot.append(np.hstack(h_plot))
plot = np.vstack(plot)
cv2.imwrite(str(output_path / f'{os.path.basename(stylization_dir)}.png'), plot)
