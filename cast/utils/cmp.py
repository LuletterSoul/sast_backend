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

style_dir = 'images/styles_0705'
content_dir = 'images/contents_0705'
# style_dir = 'images/styles_0421_k'
# content_dir = 'images/contents_0421_k'
# stylizations = ['exp/0421_cw1_lw0.1_kl50', 'exp/0421_cw1_lw1_kl50', 'exp/0421_cw1_lw10_kl50', 'exp/0421_cw1_lw100_kl50',
#                 'exp/0421_cw1_lw1000_kl50', 'exp/0421_cw1_lw10000_kl50', 'exp/0421_cw1_lw100000_kl50']

# stylizations = ['exp/0421_cw1_lw10_kl50', 'exp/0421_cw1_lw20_kl50', 'exp/0421_cw1_lw30_kl50', 'exp/0421_cw1_lw40_kl50',
#                 'exp/0421_cw1_lw50_kl50', 'exp/0421_cw1_lw60_kl50', 'exp/0421_cw1_lw70_kl50', 'exp/0421_cw1_lw80_kl50',
#                 'exp/0421_cw1_lw90_kl50', 'exp/0421_cw1_lw100_kl50']
# stylizations = [['exp/0428_cw1_lw30_k1', 'exp/0428_cw1_lw30_k5', 'exp/0428_cw1_lw30_k10', 'exp/0428_cw1_lw30_k50',
#                  'exp/0428_cw1_lw30_k100', 'exp/0428_cw1_lw30_k500', 'exp/0428_cw1_lw30_k1000'],
#                 ['exp/0423_cw1_lw50_k1', 'exp/0423_cw1_lw50_k5', 'exp/0423_cw1_lw50_k10', 'exp/0423_cw1_lw50_k50',
#                  'exp/0423_cw1_lw50_k100', 'exp/0423_cw1_lw50_k500', 'exp/0423_cw1_lw50_k1000']]
# stylizations = [['exp/0428_cw1_lw30_nups', 'exp/0428_cw1_lw30_ups200', 'exp/0428_cw1_lw30_ups100',
#                  'exp/0428_cw1_lw30_ups50', 'exp/0428_cw1_lw30_ups40',
#                  'exp/0428_cw1_lw30_ups30',
#                  'exp/0428_cw1_lw30_ups20', 'exp/0428_cw1_lw30_ups10', 'exp/0428_cw1_lw30_ups1']]
# stylizations = [
#     ['exp/0428_cw1_lw0.1_nups', 'exp/0428_cw1_lw1_nups', 'exp/0428_cw1_lw10_nups', 'exp/0428_cw1_lw100_nups',
#      'exp/0428_cw1_lw1000_nups', 'exp/0428_cw1_lw10000_nups', 'exp/0428_cw1_lw10000_nups']]


# stylizations = [
#     ['exp/0429_cw0_lw1_nups', 'exp/0429_cw0_lw1_50ups', 'exp/0429_cw0_lw1_40ups', 'exp/0429_cw0_lw1_30ups'
#         , 'exp/0429_cw0_lw1_20ups'
#         , 'exp/0429_cw0_lw1_10ups']]
# stylizations = [
#     ['exp/0429_cw0_lw1','exp/0429_cw0_lw1000']]
# stylizations = [
#     ['exp/0705_gatys', 'exp/0705_cw1_lw1_sw0_k1', 'exp/0705_cw1_lw1_ups50', 'exp/0705_cw1_lw1_sw0.001_k1',
#      'exp/0705_cw1_lw1_sw0.002_k1', 'exp/0705_cw1_lw1_sw0.003_k1', 'exp/0705_cw1_lw1_sw0.01_k1',
#      'exp/0705_cw1_lw1_sw0.05_k1', 'exp/0705_cw1_lw1_sw0.1_k1']]

stylizations = [
    ['exp/0705_gatys', 'exp/0705_cw1_lw1_sw0_k1', 'exp/0705_cw1_lw1_ups50', 'exp/0705_cw1_lw0.1_sw0.002_k1',
     'exp/0705_cw1_lw0.5_sw0.002_k1', 'exp/0705_cw1_lw10_sw0.002_k1']
]
# stylizations = [
# ['exp/0428_cw1_lw0.001','exp/0428_cw1_lw0.01','exp/0428_cw1_lw0.1', 'exp/0428_cw1_lw0.3', 'exp/0428_cw1_lw0.5',
#  'exp/0428_cw1_lw0.7', 'exp/0428_cw1_lw0.9', 'exp/0428_cw1_lw1_nups']]

# stylizations = ['exp/0427_cw1_lw30_k50_ups50_it500',
#                 'exp/0427_cw1_lw30_k50_ups50_it400', 'exp/0427_cw1_lw30_k50_ups50_it300', 'exp/0427_cw1_lw30_k50_ups50_it200',
#                 'exp/0427_cw1_lw30_k50_ups50_it100', 'exp/0427_cw1_lw30_k50_ups50_it50', 'exp/0427_cw1_lw30_k50_ups50_it40',
#                 'exp/0427_cw1_lw30_k50_ups50_it30', 'exp/0427_cw1_lw30_k50_ups50_it20']
# output_path = Path('output/contents_0427_it_cmp')
# output_path = Path('output/contents_0429_cw0_lw1-1000_cmp')
# output_path = Path('outpu1/contents_0429_cw0_lw1_ups_cmp')
output_path = Path('output/0705_ST_cmp_2')

if not output_path.exists():
    output_path.mkdir(exist_ok=True, parents=True)

style_names = os.listdir(style_dir)
content_names = os.listdir(content_dir)

for c in content_names:
    plot = []
    c_idx, extention = os.path.splitext(c)
    content_img = cv2.imread(os.path.join(content_dir, c))
    h_plot = [content_img]
    for i in range(len(stylizations[0])):
        h_plot.append(np.ones((512, 512, 3), dtype=np.uint8) * 255)
    plot.append(np.hstack(h_plot))
    for s in style_names:
        for sty in stylizations:
            s_idx, extention = os.path.splitext(s)
            h_plot = [cv2.imread(os.path.join(style_dir, s))]
            for o_path in sty:
                print(os.path.join(o_path, f'{c_idx}-{s_idx}.png'))
                output = cv2.imread(os.path.join(o_path, f'{c_idx}-{s_idx}.png'))
                if output is None:
                    raise Exception('Empty stylization loaded.')
                h_plot.append(output)
            plot.append(np.hstack(h_plot))
    plot = np.vstack(plot)
    cv2.imwrite(str(output_path / f'{c_idx}-cmp.png'), plot)
