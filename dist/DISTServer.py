# coding=UTF-8
import os
import torch
import time
from PIL import Image
from libs.Loader import Dataset, Dataset_Video
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from libs.Matrix_se_sd_new1 import MulLayer as MulLayer_se_sd_new1
from libs.models import SmallEncoder4_16x_aux, SmallDecoder4_16x
from libs.utils import makeVideo
from config.config import Config


class DISTServer(object):
    def __init__(self):
        self.content_dir = Config.content_dir_dist
        self.style_dir = Config.style_dir_dist
        self.output_dir = Config.output_dir_dist
        self.encoder_dir = Config.DIST_ENCODER
        self.decoder_dir = Config.DIST_DECODER
        self.matrix = Config.DIST_MATRIX
        self.gpu_id = Config.DIST_DEVICES
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.style_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def init_models(self):
        if torch.cuda.is_available() and self.gpu_id >= 0:
            torch.cuda.set_device(self.gpu_id)
        print(
            f'[DIST]: # CUDA:{self.gpu_id} available: {torch.cuda.get_device_name(self.gpu_id)}')
        self.enc = SmallEncoder4_16x_aux(self.encoder_dir)
        self.dec = SmallDecoder4_16x(self.decoder_dir)
        self.matrix = MulLayer_se_sd_new1('r41')

    def process(self, content_img_id, style_img_id, content_landmark, style_landmark):
        """
        :param content_img_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :return: 风格化图片id,带后缀
        """
        c_path = os.path.join(self.content_dir, content_img_id)
        s_path = os.path.join(self.style_dir, style_img_id)

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
                    stylized_img_id = self.process(
                        content_img_id, style_img_id, content_landmark, style_landmark)
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
