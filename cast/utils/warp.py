#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: warp.py
@time: 10/3/19 4:58 PM
@version 1.0
@desc:
"""

import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp


def warp_image(image, src_points=None, dst_points=None, transform=None):
    if transform is None:
        if src_points is not None and dst_points is not None:
            transform = get_transform(image, src_points, dst_points)
        else:
            raise Exception('Src points and dst points must not be None.')
    warped = warp(image, transform, output_shape=image.shape)
    return warped, transform


def get_transform(image, src_points, dst_points):
    src_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + src_points.tolist()
    )
    dst_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + dst_points.tolist()
    )
    tform3 = PiecewiseAffineTransform()
    tform3.estimate(dst_points, src_points)
    return tform3
