#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: mast_style_transfer.py
@time: 2020/8/14 20:38
@version 1.0
@desc:
"""
import os
import torch
import time
import traceback
from multiprocessing import Queue
from typing import List

import cv2 as cv
import numpy as np
from PIL import Image
from celery import shared_task
from torchvision import transforms
from torchvision.utils import save_image

from mast.libs.MAST import MAST
from mast.libs.models import Encoder, Decoder
from workers.stream import ManagedModel


@shared_task()
def mast_service():
    pass


class MastModel(ManagedModel):

    def __init__(self, gpu_id=None):
        super().__init__(gpu_id)

    #     super().__init__(gpu_id)

    def init_model(self, *args, **kwargs):
        """
        :param args:
        args[0] root: model work directroy
        args[1] cfg: model configuration
        args[2] content_dir: content dataset
        args[3] style_dir: style dataset
        args[4] stylized_dir: stylization dataset
        :param kwargs:
        :return:
        """
        self.root = args[0]
        self.cfg = args[1]
        self.content_dir = args[2]
        self.style_dir = args[3]
        self.stylized_dir = args[4]
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.style_dir, exist_ok=True)
        os.makedirs(self.stylized_dir, exist_ok=True)

        if torch.cuda.is_available() and self.cfg.gpu >= 0:
            self.cfg.device = torch.device(f'cuda')
            print(f'[Mast]: # init from CUDA device')
        else:
            self.cfg.device = 'cpu'

        self.model = MAST(self.cfg)
        decoders_path = {
            'r11': os.path.join(self.root, self.cfg.decoder_r11_path),
            'r21': os.path.join(self.root, self.cfg.decoder_r21_path),
            'r31': os.path.join(self.root, self.cfg.decoder_r31_path),
            'r41': os.path.join(self.root, self.cfg.decoder_r41_path),
            'r51': os.path.join(self.root, self.cfg.decoder_r51_path)
        }

        # set the model
        print(f'[MastService]: Loading models...')
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(os.path.join(self.root, self.cfg.encoder_path)))
        if self.cfg.type == 64:
            self.encoder = self.encoder.double()
        self.encoder = self.encoder.to(self.cfg.device)
        self.decoders = {}
        for layer_name in self.cfg.layers.split(','):
            decoder = Decoder(layer=layer_name)
            decoder.load_state_dict(torch.load(decoders_path[layer_name]))
            if self.cfg.type == 64:
                decoder = decoder.double()
            decoder = decoder.to(self.cfg.device)
            self.decoders[layer_name] = decoder
        print(f'[Mast]: Load models completely!')

    def process(self, content_img_id, style_img_id, width, height, c_mask, s_mask):
        """
        :param content_img_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :param width:
        :param height:
        :param c_mask: [height, width]
        :param s_mask: [height, width]
        :return: 风格化图片id,带后缀
        """
        import torch
        c_path = os.path.join(self.content_dir, content_img_id)
        s_path = os.path.join(self.style_dir, style_img_id)
        # print(c_path)
        # print(s_path)
        c_tensor = transforms.ToTensor()(Image.open(c_path).resize((width, height)).convert('RGB')).unsqueeze(0)
        s_tensor = transforms.ToTensor()(Image.open(s_path).resize((width, height)).convert('RGB')).unsqueeze(0)
        if self.cfg.type == 64:
            c_tensor = c_tensor.double()
            s_tensor = s_tensor.double()
        c_tensor = c_tensor.to(self.cfg.device)
        s_tensor = s_tensor.to(self.cfg.device)
        with torch.no_grad():
            sf = self.encoder(s_tensor)
        for layer_name in self.cfg.layers.split(','):
            with torch.no_grad():
                cf = self.encoder(c_tensor)[layer_name]
                csf = self.model.transform(cf, sf[layer_name], c_mask, s_mask)
                csf = self.cfg.style_weight * csf + (1 - self.cfg.style_weight) * cf
                out_tensor = self.decoders[layer_name](csf, layer_name)
            c_tensor = out_tensor
        stylized_img_id = f'{os.path.splitext(content_img_id)[0]}_{os.path.splitext(style_img_id)[0]}.png'
        out_path = os.path.join(self.stylized_dir, stylized_img_id)
        save_image(out_tensor, out_path, nrow=1, padding=0)
        return stylized_img_id

    @staticmethod
    def convert_points(width, height, points_list):
        width_offset = int(width / 2)
        height_offset = int(height / 2)
        for i in range(len(points_list)):
            for j in range(len(points_list[i])):
                points_list[i][j][0] = int(points_list[i][j][0] + width_offset)
                points_list[i][j][1] = int(points_list[i][j][1] + height_offset)
        return points_list

    def create_content_and_style_mask(self, width, height, content_mask_points_list, style_mask_points_list):
        if content_mask_points_list == [] or style_mask_points_list == []:
            c_mask = None
            s_mask = None
        else:
            content_mask_points_list = self.convert_points(width, height, content_mask_points_list)
            style_mask_points_list = self.convert_points(width, height, style_mask_points_list)
            # print(f'content_mask_points_list={content_mask_points_list}')
            # print(f'style_mask_points_list={style_mask_points_list}')
            c_mask = np.zeros((height, width), dtype=np.uint8)
            s_mask = np.zeros((height, width), dtype=np.uint8)
            label = 1
            for c_mask_points in content_mask_points_list:
                c_mask_points_array = [np.array(c_mask_points)]
                c_mask = cv.fillPoly(c_mask, c_mask_points_array, label)
                label += 1
            label = 1
            for s_mask_points in style_mask_points_list:
                s_mask_points_array = [np.array(s_mask_points)]
                s_mask = cv.fillPoly(s_mask, s_mask_points_array, label)
                label += 1
        return c_mask, s_mask

    def predict(self, msg):
        print(f'[Mast]: get msg {msg} from receive queue, start process...')
        req_id = msg.get('req_id')
        content_img_id = msg.get('content_id')
        style_img_id = msg.get('style_id')
        width = msg.get('width')
        height = msg.get('height')
        content_mask_points_list = msg.get('content_mask')
        style_mask_points_list = msg.get('style_mask')
        c_mask, s_mask = self.create_content_and_style_mask(width, height, content_mask_points_list,
                                                            style_mask_points_list)
        try:
            s = time.time()
            stylized_img_id = self.process(content_img_id, style_img_id, width, height, c_mask, s_mask)
            t = time.time() - s
            print(f'Consuming time {round(t, 4)}')
            result_msg = {
                'req_id': req_id,
                'content_img_id': content_img_id,
                'style_img_id': style_img_id,
                'stylized_img_id': stylized_img_id,
                'process_step': -1,
                'status': 'success'
            }
            # results_queue.put(result_msg)
            print(f'[Mast]: result msg have put into results queue...')
            return result_msg
        except Exception as e:
            traceback.print_exc()
            print(f'[Mast]: MAST exception: {e}')
            stylization_id = f'{os.path.splitext(content_img_id)[0]}_{os.path.splitext(style_img_id)[0]}.png'
            result_msg = {
                'req_id': req_id,
                'content_img_id': content_img_id,
                'style_img_id': style_img_id,
                'stylized_img_id': stylization_id,
                'process_step': -1,
                'status': 'failed'
            }
            return result_msg
            # results_queue.put(result_msg)
