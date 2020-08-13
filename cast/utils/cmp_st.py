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

style_dir = 'images/styles_104'
content_dir = 'images/contents_total_0425'
method_dir = '/Users/luvletteru/0516-Style-Transfer'

stylizations = [
    ['exp/0428_cw1_lw0.1', 'exp/0428_cw1_lw0.2', 'exp/0428_cw1_lw0.3', 'exp/0428_cw1_lw0.4', 'exp/0428_cw1_lw0.5',
     'exp/0428_cw1_lw0.6', 'exp/0428_cw1_lw0.7', 'exp/0428_cw1_lw0.8', 'exp/0428_cw1_lw0.9', 'exp/0428_cw1_lw1_nups']]

output_path = Path('output/ST_cmp')

methods = ['Adain', 'LST', 'MST', 'StyleSwap', 'WCT', 'CAST']
methods = [os.path.join(method_dir, m) for m in methods]

if not output_path.exists():
    output_path.mkdir(exist_ok=True, parents=True)

person_names = os.listdir(os.path.join(method_dir, 'CAST'))
content_names = os.listdir(content_dir)
style_names = os.listdir(style_dir)

for c in content_names:
    plot = []
    c_idx, extention = os.path.splitext(c)
    content_img = cv2.imread(os.path.join(content_dir, c))
    c_idx = c_idx.replace(' ', '_')
    h_plot = [content_img]
    for i in range(len(methods)):
        h_plot.append(np.ones((512, 512, 3), dtype=np.uint8) * 255)
    content_plot = np.hstack(h_plot)
    plot.append(content_plot)
    person_output = output_path / c_idx
    person_output.mkdir(exist_ok=True, parents=True)
    for idx, s in enumerate(style_names):
        write_dir = str(person_output / f'{c_idx}-{idx}-cmp.png')
        if os.path.exists(write_dir):
            print(f'Exisit in {write_dir}')
            continue
        s_idx, extention = os.path.splitext(s)
        style_img_path = os.path.join(style_dir, s)
        style = cv2.imread(style_img_path)
        if style is None:
            raise Exception(f'Empty style image: {style_img_path}')
        h_plot = [style]
        for m in methods:
            print(os.path.join(m, c_idx, f'{c_idx}-{s_idx}.png'))
            output = cv2.imread(os.path.join(m, c_idx, f'{c_idx}-{s_idx}.png'))
            if output is None:
                raise Exception('Empty stylization loaded.')
            h_plot.append(output)
        plot.append(np.hstack(h_plot))
        if (idx + 1) % 10 == 0:
            plot = np.vstack(plot)
            cv2.imwrite(write_dir, plot)
            plot = [content_plot]
    if len(plot):
        plot = np.vstack(plot)
        cv2.imwrite(str(person_output / f'{c_idx}-end-cmp.png'), plot)
