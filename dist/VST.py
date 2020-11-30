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

################### 参数定义 ##################
encoder_dir = 'models/4SE.pth'
decoder_dir = 'models/4SD.pth'
matrix_dir = 'models/FTM.pth'
style_dir = "data/style/"
content_dir = "data/video/"
loadSize = 512
fineSizeH = 512
fineSizeW = 1024
outf = f'Video_Results' 

os.environ['CUDA_VISIBLE_DEVICES']='0'

#################  #################
cuda = torch.cuda.is_available()
os.makedirs(outf,exist_ok=True) 
cudnn.benchmark = True

################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((fineSizeH, fineSizeW)),
                transforms.ToTensor()])
    return transform(img)

style_dataset = Dataset_Video(style_dir,loadSize,fineSizeH, fineSizeW, test=True, video=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                           batch_size = 1,
                                           shuffle = False,
                                           num_workers = 1)

enc = SmallEncoder4_16x_aux(encoder_dir)
dec = SmallDecoder4_16x(decoder_dir)

matrix = MulLayer_se_sd_new1('r41')
matrix.load_state_dict(torch.load(matrix_dir, map_location='cuda:0'))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(1,3,fineSizeH,fineSizeW) 
styleV = torch.Tensor(1,3,fineSizeH,fineSizeW)

################# GPU  #################
if(cuda):
    enc.cuda()
    dec.cuda()
    matrix.cuda()
    styleV = styleV.cuda()
    contentV = contentV.cuda()

video_list = [x for x in os.listdir(content_dir) if 'DS' not in x]
video_list = sorted(video_list)

for sj,(style,styleName) in enumerate(style_loader):
    styleName = styleName[0]
    styleV.resize_(style.size()).copy_(style)
    sF = enc(styleV)
    style = styleV.squeeze(0).cpu().numpy()

    for ci,video_name in enumerate(video_list):
        now_video_dir = os.path.join(content_dir,video_name)
        # print(now_video_dir)
        content_dataset = Dataset_Video(now_video_dir, loadSize,fineSizeH, fineSizeW, test=True, video=True)
        content_loader = torch.utils.data.DataLoader(dataset    = content_dataset,
                                                batch_size = 1,
                                                shuffle    = False)     
        result_frames = []
        contents = []
        print('Transfer %s with %s...'%(video_name, styleName))
        start_time = time.time()

        frames_dir = f'{outf}/{video_name}_{styleName}_frames'
        os.makedirs(frames_dir ,exist_ok=True)

        for i,(content,contentName) in enumerate(content_loader):
            print('Transfer frame %d...'%i)
            contentName = contentName[0]
            if(cuda):
                contentV = contentV.cuda()
            contentV.resize_(content.size()).copy_(content)
            contents.append(content.squeeze(0).float().numpy())
            # forward
            with torch.no_grad():
                cF = enc(contentV)
                feature,transmatrix = matrix(cF, sF)
                transfer = dec(feature)

            transfer = transfer.clamp(0,1)
            result_frames.append(transfer.squeeze(0).cpu().numpy())

            vutils.save_image(transfer,'%s/%d.png'%(frames_dir,i+1))

        end_time = time.time()
        print('Elapsed time is: %.4f seconds' % (end_time - start_time))
        makeVideo(contents,style,result_frames,outf,video_name,styleName)