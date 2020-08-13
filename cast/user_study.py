#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: user_study.py
@time: 5/20/20 10:47 AM
@version 1.0
@desc:
"""
from pathlib import Path
import cv2
import os
from random import sample

import imutils

content_dir = 'data/contents_total_0425'

photo_dir = 'datasets/WebCari_512/img'

CAST_dir = '/Users/luvletteru/0516-Style-Transfer/CAST'
WarpGAN_dir = '/Users/luvletteru/Documents/0427-WarpGAN'

output_dir = 'output/user_study'

content_names = os.listdir(content_dir)
from multiprocessing import Process




def from_CAST():
    for c in content_names:
        names = c.split('_')[:-1]
        person_name = names[0]
        for n in names[1:]:
            person_name += ' ' + n
        print(f'Person Name: [{person_name}]')
        photo_list = list((Path(photo_dir) / person_name).glob('P*'))
        cari_paths = sample(list((Path(photo_dir) / person_name).glob('C*')), 1)
        if not len(photo_list):
            raise Exception('Empty photo list.')
        if len(photo_list) >= 3:
            photo_paths = sample(photo_list, 3)
        else:
            photo_paths = photo_list
            diff = 3 - len(photo_paths)
            for i in range(diff):
                photo_paths.append(photo_paths[-1])
        print(str(Path(CAST_dir) / c[:-4].replace(' ', '_')))
        cast_cari_paths = sample(list((Path(CAST_dir) / c[:-4].replace(' ', '_')).glob('*')), 1)
        prefix = c[:-4].replace(' ', '_')
        print(prefix)
        warp_cari_paths = sample(list(Path(WarpGAN_dir).glob(f'{prefix}*')), 1)
        cast_folder = person_name.replace(' ', '_') + '_' + os.path.basename(CAST_dir)
        warp_gan_folder = person_name.replace(' ', '_') + '_' + os.path.basename(WarpGAN_dir)
        cari_folder = person_name.replace(' ', '_', ) + '_hand'
        cast_output_path = os.path.join(output_dir, cast_folder)
        warp_gan_output_path = os.path.join(output_dir, warp_gan_folder)
        cari_output_path = os.path.join(output_dir, cari_folder)
        if not os.path.exists(cast_output_path):
            Path(cast_output_path).mkdir(exist_ok=True, parents=True)
            print(f'Created {cast_output_path}')
        if not os.path.exists(warp_gan_output_path):
            Path(warp_gan_output_path).mkdir(exist_ok=True, parents=True)
            print(f'Created {warp_gan_output_path}')
        if not os.path.exists(cari_output_path):
            Path(cari_output_path).mkdir(exist_ok=True, parents=True)
            print(f'Created {cari_output_path}')
        for p in photo_paths:
            photo = cv2.imread(str(p))
            photo_name = os.path.basename(str(p))
            cast_output_img = os.path.join(cast_output_path, photo_name)
            warp_gan_output_img = os.path.join(warp_gan_output_path, photo_name)
            cari_output_img = os.path.join(cari_output_path, photo_name)
            cv2.imwrite(cast_output_img, photo)
            print(f'Write image into {cast_output_img}')
            cv2.imwrite(warp_gan_output_img, photo)
            print(f'Write image into {warp_gan_output_img}')
            cv2.imwrite(cari_output_img, photo)
            print(f'Write image into {cari_output_img}')
        for idx, cp in enumerate(cast_cari_paths):
            cari = cv2.imread(str(cp))
            output_img = os.path.join(cast_output_path, f'C_{idx}.png')
            cv2.imwrite(output_img, cari)
        for idx, wp in enumerate(warp_cari_paths):
            cari = cv2.imread(str(wp))
            cari = imutils.resize(cari, width=512, inter=cv2.INTER_CUBIC)
            output_img = os.path.join(warp_gan_output_path, f'C_{idx}.png')
            # output_img = os.path.join(warp_gan_output_path, cari_name)
            cv2.imwrite(output_img, cari)
        for idx, cp in enumerate(cari_paths):
            cari = cv2.imread(str(cp))
            output_img = os.path.join(cari_output_path, f'C_{idx}.png')
            cv2.imwrite(output_img, cari)


if __name__ == '__main__':
    from_CAST()
