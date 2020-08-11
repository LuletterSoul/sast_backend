# coding=UTF-8
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from multiprocessing import Queue
from PIL import Image
import sys
import time

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
from ..libs.models import Encoder, Decoder
from ..libs.MAST import MAST
from ..libs.MastConfig import MastConfig
from config.config import Config


class MastServer(object):
    def __init__(self):
        self.root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
        self.cfg = MastConfig(os.path.join(self.root, f'configs/MAST_Configs.yml'))
        self.content_dir = Config.CONTENT_DIRECTORY
        self.style_dir = Config.STYLE_DIRECTORY
        self.stylized_dir = Config.STYLIZATION_DIRECTORY
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.style_dir, exist_ok=True)
        os.makedirs(self.stylized_dir, exist_ok=True)

    def init_models(self):
        self.mast = MAST(self.cfg)
        if torch.cuda.is_available() and self.cfg.gpu >= 0:
            self.cfg.device = torch.device(f'cuda:{self.cfg.gpu}')
            print(f'[Mast]: # CUDA:{self.cfg.gpu} available: {torch.cuda.get_device_name(self.cfg.gpu)}')
        else:
            self.cfg.device = 'cpu'

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

    def process(self, content_img_id, style_img_id):
        """
        :param content_img_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :return: 风格化图片id,带后缀
        """
        c_path = os.path.join(self.content_dir, content_img_id)
        s_path = os.path.join(self.style_dir, style_img_id)
        c_tensor = transforms.ToTensor()(Image.open(c_path).convert('RGB')).unsqueeze(0)
        s_tensor = transforms.ToTensor()(Image.open(s_path).convert('RGB')).unsqueeze(0)
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
                csf = self.mast.transform(cf, sf[layer_name])
                csf = self.cfg.style_weight * csf + (1 - self.cfg.style_weight) * cf
                out_tensor = self.decoders[layer_name](csf, layer_name)
            c_tensor = out_tensor
        stylized_img_id = f'{os.path.splitext(content_img_id)[0]}_{os.path.splitext(style_img_id)[0]}.png'
        out_path = os.path.join(self.stylized_dir, stylized_img_id)
        save_image(out_tensor, out_path, nrow=1, padding=0)
        return stylized_img_id

    def run(self, receive_queue: Queue, results_queue: Queue):
        """
        receive_queue中存储的消息格式为
        {
            'content_img_id': content_img_id,
            'style_img_id': style_img_name
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
        self.init_models()
        while True:
            if not receive_queue.empty():
                msg = receive_queue.get()
                print(f'[Mast]: get msg from receive queue, start process...')
                content_img_id = msg['content_img_id']
                style_img_id = msg['style_img_id']
                try:
                    stylized_img_id = self.process(content_img_id, style_img_id)
                    result_msg = {
                        'content_img_id': content_img_id,
                        'style_img_id': style_img_id,
                        'stylized_img_id': stylized_img_id,
                        'process_step': -1
                    }
                    results_queue.put(result_msg)
                    print(f'[Mast]: result msg have put into results queue...')
                except Exception as e:
                    print(f'[Mast]: MAST exception: {e}')
