import os
from PIL import Image
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import re
from torch.autograd import Variable

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False,video=False):
        super(Dataset,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in os.listdir(dataPath) if is_image_file(x)]
        self.image_list = sorted(self.image_list)
        if(video):
            self.image_list = sorted(self.image_list)
        if not test:
            self.transform = transforms.Compose([
            		         transforms.Resize(fineSize),
            		         transforms.RandomCrop(fineSize),
                             transforms.RandomHorizontalFlip(),
            		         transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                            #  transforms.Resize(fineSize),
            		         transforms.Resize((fineSize, fineSize)),
            		         transforms.ToTensor()])

        self.test = test

    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0]
        return ImgA,imgName

    def __len__(self):
        return len(self.image_list)

def tryint(s):  # 将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):  # 将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):  # 以分割后的list为单位进行排序
    """
    sort list strings according string and number
    :param v_list:
    :return:
    """
    return sorted(v_list, key=str2int, reverse=False)

class Dataset_Video(data.Dataset):
    def __init__(self,dataPath,loadSize, fineSizeH, fineSizeW, test=False,video=False):
        super(Dataset_Video,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in os.listdir(dataPath) if is_image_file(x) and os.path.basename !='preview.png']
        self.image_list = sort_humanly(self.image_list)
        print(self.image_list)
        if(video):
            self.image_list = sort_humanly(self.image_list)
        if not test:
            self.transform = transforms.Compose([
            		         transforms.Resize((fineSizeH, fineSizeW)),
            		         transforms.RandomCrop(fineSizeH, fineSizeW),
                             transforms.RandomHorizontalFlip(),
            		         transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            		         transforms.Resize((fineSizeH, fineSizeW)),
            		         transforms.ToTensor()])

        self.test = test

    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0]
        return ImgA,imgName

    def __len__(self):
        return len(self.image_list)