# coding=UTF-8
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

def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((Config.fineSizeH, Config.fineSizeW)),
                transforms.ToTensor()])
    return transform(img)

class DISTModel(ManagedModel):
    def __init__(self, gpu_id=None):
        super().__init__(gpu_id)
        self.content_dir = Config.content_dir_dist
        self.style_dir = Config.style_dir_dist
        self.output_dir = Config.output_dir_dist
        self.encoder_dir = Config.encoder_dir_dist
        self.decoder_dir = Config.decoder_dir_dist
        self.matrix = Config.matrix_dir_dist
        self.gpu_id = Config.gpu_id
        os.makedirs(self.output_dir, exist_ok=True)
        
    def init_models(self):
        if torch.cuda.is_available() and self.gpu_id >= 0:
            torch.cuda.set_device(self.gpu_id)
        print(f'[DIST]: # CUDA:{self.gpu_id} available: {torch.cuda.get_device_name(self.gpu_id)}')
        self.enc = SmallEncoder4_16x_aux(self.encoder_dir)
        self.dec = SmallDecoder4_16x(self.decoder_dir)
        self.matrix = MulLayer_se_sd_new1('r41')
        self.matrix.load_state_dict(torch.load('dist/models/FTM.pth', map_location='cuda:0'))
        if(torch.cuda.is_available()):
            self.enc.cuda()
            self.dec.cuda()
            self.matrix.cuda()

    def process(self, video_dir_id, style_img_id):
        """
        :param video_dir_id: 内容图id,带后缀
        :param style_img_id: 风格图id,带后缀
        :return: 风格化视频文件目录id
        """
        print(style_img_id)
        styleV = loadImg(style_img_id).unsqueeze(0)
        content_dataset = Dataset_Video(video_dir_id, Config.loadSize, Config.fineSizeH, Config.fineSizeW, test=True, video=True)
        content_loader = torch.utils.data.DataLoader(dataset    = content_dataset,
                                                batch_size = 1,
                                                shuffle    = False)   
        contentV = torch.Tensor(1,3,Config.fineSizeH,Config.fineSizeW)
        if(torch.cuda.is_available()):
            styleV.cuda()
            contentV.cuda()
        result_frames = []
        contents = []
        style = styleV.squeeze(0).cpu().numpy()
        sF = self.enc(styleV)
        start_time = time.time()
        for i,(content,contentName) in enumerate(content_loader):
            print('Transfer frame %d...'%i)
            contentName = contentName[0]
            contentV.resize_(content.size()).copy_(content)
            contents.append(content.squeeze(0).float().numpy())
            # forward
            with torch.no_grad():
                cF = self.enc(contentV)
                feature,transmatrix = self.matrix(cF,sF)
                transfer = self.dec(feature)

            transfer = transfer.clamp(0,1)
            result_frames.append(transfer.squeeze(0).cpu().numpy())
        end_time = time.time()
        print('Elapsed time is: %.4f seconds' % (end_time - start_time))
        makeVideo(contents,style,result_frames,Config.output_dir_dist, video_dir_id.split('/')[-2], os.path.basename(style_img_id).split('.')[0])
        c = video_dir_id.split('/')[-2]
        s =  os.path.basename(style_img_id).split('.')[0]
        video_result_dir_id = f'{c}_{s}.avi'
        return video_result_dir_id

    def predict(self, msg):
        print(f'[DIST]: get msg {msg} from receive queue, start process...')
        video_dir_id = msg.get('video_dir_id')
        style_img_id = msg.get('style_id')
        c = video_dir_id.split('/')[-2]
        s =  os.path.basename(style_img_id).split('.')[0]
        video_result_id = f'{c}_{s}.avi'
        msg['video_result_id'] = f'data/Video_Results/{video_result_id}'
        msg['timestamp'] = time.time()
        try:
            self.process(video_dir_id, style_img_id)
            msg['status'] = 'success'
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


    

