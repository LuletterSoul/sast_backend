import logging
import os
import random
import re
import time

import cv2
import numpy as np
from constant import colormap

label_list = ['bg', 'skin', 'nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'mouth', 'u_lip',
              'l_lip']


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" % (
            f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__PyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    call(["cat", "/usr/local/cuda/version.txt"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('__Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('__Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('__Current cuda device  {}'.format(torch.cuda.current_device()))


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])




cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道

for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引


def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


# def colorize(semantic_image, n):
#     semantic_image = torch.squeeze(to_tensor(semantic_image))
#     # semantic_image = torch.from_numpy(semantic_image)
#     cmap = torch.from_numpy(colormap[:n])
#     H, W = semantic_image.size()
#     color_image = torch.ByteTensor(3, H, W).fill_(0)
#     for label in range(0, len(cmap)):
#         mask = (label == semantic_image)
#         color_image[0][mask] = cmap[label][0]
#         color_image[1][mask] = cmap[label][1]
#         color_image[2][mask] = cmap[label][2]
#     return color_image


def colorize(semantic_image, type='tensor'):
    tensor = torch.from_numpy(colormap[np.array(semantic_image)])
    if type == 'tensor':
        if len(tensor.size()) == 4:
            return tensor.permute(0, 3, 1, 2)
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor
    else:
        return tensor.data.cpu().numpy()


import torch
import math

irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


class AngleFactory:
    """method for angle calculation"""

    @staticmethod
    def __calAngleBetweenTwoVector(vectorA, vectorB):
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = 0
        if abs(lenA * lenB - 0) > 10e-4:
            cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        if abs(cosAngle - 1) <= 10e-4:
            return np.deg2rad(0)
        # notes that dot product calculates min angle between vectorA and vectorB only.
        angle = np.arccos(cosAngle)
        return angle

    @classmethod
    def calAngleBetweenTwoVector(cls, vectorA, vectorB):
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        angle = np.arccos(cosAngle)
        if abs(cosAngle - 1) <= 10e-4:
            return np.deg2rad(0)
        return angle

    @classmethod
    def calAngleClockwise(cls, startPoint, endPoint, centerPoint):
        """
        get clockwise angle formed by three point
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :return: clockwise angle
        """
        vectorA = startPoint - centerPoint
        vectorB = endPoint - centerPoint
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle
        # if counter-clockwise
        # if cross product(two-dim vector's cross product returns a integer only)
        # is negative ,means the normal vector is oriented down,vectorA is in the clockwise of vectorB
        # otherwise in counterclockwise.
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle
        return angle

    @classmethod
    def calAngleClockwiseByVector(cls, vectorA, vectorB):
        """
        get clockwise angle formed by two vector
        """
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle
        # if counter-clockwise
        # if cross product(two-dim vector's cross product returns a integer only)
        # is negative ,means the normal vector is oriented down,vectorA is in the clockwise of vectorB
        # otherwise in counterclockwise.
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        return angle

    @classmethod
    def calPointerValueByOuterPoint(cls, startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue):
        """
        get value of pointer meter
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param pointerPoint: pointer's outer point
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)
        angle = cls.calAngleClockwise(startPoint, pointerPoint, centerPoint)
        value = angle / angleRange * totalValue + startValue

        return value

    @classmethod
    def calPointerValueByPointerVector(cls, startPoint, endPoint, centerPoint, PointerVector, startValue, totalValue):
        """
        get value of pointer meter
        注意传入相对圆心的向量
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param PointerVector: pointer's vector
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = PointerVector

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue

        return value

    @classmethod
    def calPointerValueByPoint(cls, startPoint, endPoint, centerPoint, point, startValue, totalValue):
        """
        :param startPoint: 起点
        :param endPoint: 终点
        :param centerPoint:
        :param point:
        :param startValue:
        :param totalValue:
        :return:
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = point - centerPoint

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle

        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue

        return value

re_digits = re.compile(r'(\d+)')

def emb_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_strings_with_emb_numbers(alist):
    aux = [(emb_numbers(s), s) for s in alist]
    aux.sort()
    return [s for __, s in aux]


def sort_strings_with_emb_numbers2(alist):
    return sorted(alist, key=emb_numbers)


def get_dirs(current_dir):
    dirs = []
    for d in os.listdir(current_dir):
        jd = os.path.join(current_dir, d)
        if os.path.isdir(jd) and not d.startswith('_') and not d.startswith('.'):
            dirs.append((d, jd))

    return dirs


def sorted_names(jd):
    childs = [c for c in os.listdir(jd) if not c.startswith('.') and not c.startswith('_')]
    # childs.sort()
    childs = sort_strings_with_emb_numbers(childs)
    return childs


def get_filenames(path, offset=None):
    if offset is None:
        face_filenames = sorted_names(path)
    else:
        print('Offset: [{}]'.format(offset))
        face_filenames = sorted_names(path)[:offset]
    return face_filenames


def draw_key_points(src, key_points):
    img = src.copy()
    for pt in key_points:
        # fpts = f_kl[index]
        # cpts = c_kl[index]
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        if is_filtered(pt):
            continue
        cv2.circle(img, center=(int(pt[0]), int(pt[1])), radius=2, color=(b, g, r), thickness=-1)
    return img


def is_filtered(pt):
    return pt[0] == -1 and pt[1] == -1
