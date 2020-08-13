#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: _rename.py
@time: 5/17/20 4:11 PM
@version 1.0
@desc:
"""
import os
from pathlib import Path

import cv2
import numpy as np

input_dir = '/Users/luvletteru/0516-Style-Transfer/MST'

output_dir = '/Users/luvletteru/0516-Style-Transfer1/MST'

Path(output_dir).mkdir(exist_ok=True, parents=True)

for per_name in os.listdir(input_dir):
    for photo_name in os.listdir(os.path.join(input_dir, per_name)):
        if not os.path.exists(os.path.join(output_dir, per_name)):
            os.mkdir(os.path.join(output_dir, per_name))
        split_res = photo_name.split('.')
        img = cv2.imread(os.path.join(input_dir, per_name, photo_name))
        cv2.imwrite(os.path.join(output_dir, per_name, f'{split_res[0]}.png'), img)
