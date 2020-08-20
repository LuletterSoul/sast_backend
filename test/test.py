#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test.py
@time: 2020/8/18 12:05
@version 1.0
@desc:
"""

import numpy as np

if __name__ == '__main__':
    path = '/Users/shandalau/Documents/Github/sast_backend/data/landmarks/Amedeo_Modigliani_2.txt'
    landmark = np.loadtxt(path)
    landmark = landmark * 2
    np.savetxt(path, landmark,
               fmt='%d')
