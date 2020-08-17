# coding=UTF-8
import os
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
from torchvision.utils import save_image
from multiprocessing import Queue
from PIL import Image
from config.config import Config
from .cast import warp_content_to_style_datasets, warp_content_to_style_images
from .sast import st_content_to_style

class CastServer(object):
    def __init__(self):
        self.content_dir = Config.content_dir
        self.style_dir = Config.style_dir
        self.landmark_dir = Config.landmark_dir
        self.output_dir = Config.output_dir
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.style_dir, exist_ok=True)
        os.makedirs(self.landmark_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, content_img_id, style_img_id, content_landmark, style_landmark):
        """
        :param content_img_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :return: 风格化图片id,带后缀
        """
        c_path = os.path.join(self.content_dir, content_img_id)
        s_path = os.path.join(self.style_dir, style_img_id)
        warp_content_to_style_images(photo_content = c_path,caricature_style = s_path)
        stylized_img_id = st_content_to_style()
        return stylized_img_id

    def run(self, receive_queue1: Queue, results_queue1: Queue):
        """
        receive_queue中存储的消息格式为
        {
                'content_img_id': content_id,
                'style_img_id': style_id,
                'content_landmark': content_landmark,
                'style_landmark': style_landmark
        }
        其中id为string格式，带有后缀，如in1.png
        results_queue中存储的消息格式为
        {
            'content_img_id': content_img_id,
            'style_img_id': style_img_id,
            'stylized_img_id': stylized_img_id
            'process_step': -1
        }
        其中id为string格式
        :param receive_queue:
        :param results_queue:
        :return:
        """
        while True:
            if not receive_queue1.empty():
                msg = receive_queue1.get()
                print(f'[Mast]: get msg from receive queue, start process...')
                content_img_id = msg['content_img_id']
                style_img_id = msg['style_img_id']
                content_landmark = msg['content_landmark']
                style_landmark = msg['style_landmark']
                try:
                    stylized_img_id = self.process(content_img_id, style_img_id, content_landmark, style_landmark)
                    result_msg = {
                        'content_img_id': content_img_id,
                        'style_img_id': style_img_id,
                        'stylized_img_id': stylized_img_id,
                        'process_step': -1
                    }
                    results_queue1.put(result_msg)
                    print(f'[Cast]: result msg have put into results queue...')
                except Exception as e:
                    print(f'[Cast]: CAST exception: {e}')