# coding=UTF-8
from api import category
import os
import torch
import time
from PIL import Image
import traceback
from celery import shared_task
from dist.libs.Loader import Dataset, Dataset_Video
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from dist.libs.Matrix_se_sd_new1 import MulLayer as MulLayer_se_sd_new1
from dist.libs.models import SmallEncoder4_16x_aux, SmallDecoder4_16x
from dist.libs.utils import makeVideo
from config.config import Config
from sockets import *
from workers.stream import ManagedModel, mp, threading, run_redis_workers_forever
from utils import parse_devices
from sockets import *


def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((Config.FINE_SIZE_H, Config.FINE_SIZE_W)),
        transforms.ToTensor()])
    return transform(img)


class DISTModel(ManagedModel):
    def __init__(self, gpu_id=None):
        super().__init__(gpu_id)
        self.content_dir = Config.CONTENT_DIRECTORY
        self.style_dir = Config.STYLE_DIRECTORY
        self.output_dir = Config.STYLIZATION_DIRECTORY
        self.encoder_dir = Config.DIST_ENCODER
        self.decoder_dir = Config.DIST_DECODER
        self.matrix = Config.DIST_MATRIX
        os.makedirs(self.output_dir, exist_ok=True)

    def init_model(self):
        # if torch.cuda.is_available() and self.gpu_id >= 0:
        # torch.cuda.set_device(self.gpu_id)
        # print(
        # f'[DIST]: # CUDA:{self.gpu_id} available: {torch.cuda.get_device_name(self.gpu_id)}')
        print(f'[DIST]: # init from CUDA device')
        self.enc = SmallEncoder4_16x_aux(self.encoder_dir)
        self.dec = SmallDecoder4_16x(self.decoder_dir)
        self.matrix = MulLayer_se_sd_new1('r41')
        self.matrix.load_state_dict(torch.load(
            'dist/models/FTM.pth', map_location='cuda:0'))
        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.matrix = self.matrix.cuda()
        print(f'[DIST]: Load models completely!')

    def process(self, content_path, style_path,content_name,style_name):
        """
        :param video_dir_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :return: 风格化视频文件目录id
        """
        print(style_path)
        styleV = loadImg(style_path).unsqueeze(0)
        content_dataset = Dataset_Video(
            content_path, Config.LOAD_SIZE, Config.FINE_SIZE_H, Config.FINE_SIZE_W, test=True, video=True)
        content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                     batch_size=1,
                                                     shuffle=False)
        contentV = torch.Tensor(1, 3, Config.FINE_SIZE_H, Config.FINE_SIZE_W)
        if torch.cuda.is_available():
           styleV = styleV.cuda()
           contentV = contentV.cuda()
        result_frames = []
        contents = []
        style = styleV.squeeze(0).cpu().numpy()
        sF = self.enc(styleV)
        start_time = time.time()
        # body = {
        #     'sid': sid,
        #     'req_id': req_id,
        #     'content_id': req['content_id'],
        #     'style_id': req['style_id'],
        #     'stylization_id': stylization_id,
        #     'current_update_steps': -1,
        #     'current_cost_time': cost_time,
        #     'percent': round(cost_time / Config.MAST_TOTAL_TIME * 100, 1),
        #     # 1 represent 'COMPLETE',otherwise it is 'SYNTHESISING',
        #     'total_time': Config.MAST_TOTAL_TIME,
        #     'total_update_steps': -1,
        # }
        for i, (content, contentName) in enumerate(content_loader):
            print('Transfer frame %d...' % i)
            contentName = contentName[0]
            contentV.resize_(content.size()).copy_(content)
            contents.append(content.squeeze(0).float().numpy())
            # forward
            with torch.no_grad():
                cF = self.enc(contentV)
                feature, transmatrix = self.matrix(cF, sF)
                transfer = self.dec(feature)

            transfer = transfer.clamp(0, 1)
            result_frames.append(transfer.squeeze(0).cpu().numpy())
        end_time = time.time()
        print('Elapsed time is: %.4f seconds' % (end_time - start_time))
        stylization_id = makeVideo(f'{content_path}.mp4',contents, style, result_frames, self.output_dir, content_name, style_name)
        # c = content_path.split('/')[-2]
        # s = os.path.basename(style_path).split('.')[0]
        # video_result_dir_id = f'{c}_{s}.avi'
        # return video_result_dir_id
        return stylization_id

    def predict(self, msg):
        print(f'[DIST]: get msg {msg} from receive queue, start process...')
        content_id = msg.get('content_id')
        style_id = msg.get('style_id')
        content_name = os.path.splitext(content_id)[0]
        style_name = os.path.splitext(style_id)[0]
        category = msg.get('category')
        style_category = msg.get('style_category')
        content_category = msg.get('content_category')
        content_path = os.path.join(
            Config.CONTENT_DIRECTORY, content_category, content_name)
        style_path = os.path.join(
            category, Config.STYLE_DIRECTORY,  style_category,style_id)
        msg['timestamp'] = time.time()
        try:
            stylization_id = self.process(content_path, style_path, content_name, style_name)
            msg['status'] = 'success'
            msg['stylization_id'] = stylization_id
            synthesis_complete(msg)
            print(f'[DIST]: result msg have put into results queue...')
            return msg
        except Exception as e:
            traceback.print_exc()
            print(f'[DIST]: DIST exception: {e}')
            msg['status'] = 'failed'
            synthesis_failed(msg)
            return msg
        finally:
            pass


def create_dist_worker():

    devices = parse_devices(Config.DIST_DEVICES)
    destroy_event = mp.Event()
    # batch_size = Config.MAST_BATCH_SIZE
    # worker_num = Config.MAST_WORKER_NUM
    thread = threading.Thread(target=run_redis_workers_forever, args=(DISTModel, Config.DIST_BATCH_SIZE, 0.1,
                                                                      Config.DIST_WORKER_NUM, devices,
                                                                      Config.REDIS_BROKER_URL, Config.DIST_CHANNEL,
                                                                      (), None, destroy_event,),
                              daemon=True)
    thread.start()
    return thread, destroy_event
