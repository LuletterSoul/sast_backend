#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: warp.py
@time: 10/3/19 4:58 PM
@version 1.0
@desc:
"""
import logging
import os
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from  skimage.transform import PiecewiseAffineTransform, warp

from config.config import setup_logging, DEBUG
from constant import *
from utils.misc import label_list, AngleFactory, image2label
from utils.transforms import ToUnNormalizedTensor

logger_name = 'warp_logger'
level = logging.INFO
logger = setup_logging('.', logger_name, level)

# CARI_IMG_PATH = '../datasets/Caricature-img'
# FACE_IMG_PATH = '../datasets/CelebA-HQ-img'
# CARI_DATASET_PATH = '../datasets/Caricature-mask'
# FACE_DATASET_PATH = '../datasets/CelebAMaskHQ-mask'
# CARI_DATASET_COLOR_PATH = '../datasets/Caricature-mask-color'
# FACE_DATASET_COLOR_PATH = '../datasets/CelebAMaskHQ-mask-color'
# FACE_WARPED = '../datasets/CelebA-HQ-img-Warped'

face_img_name = '1.png'
cari_img_name = '1'
face_mask_path = os.path.join(FACE_MASK_PATH, face_img_name)
face_path = os.path.join(FACE_IMG_PATH, '1.jpg')
cari_mask_path = os.path.join(CARI_MASK_PATH, cari_img_name + '.png')
cari_path = os.path.join(CARI_IMG_PATH, cari_img_name + '.jpg')
face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
cari_mask = cv2.imread(cari_mask_path, cv2.IMREAD_GRAYSCALE)
# 'skin', 'nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'mouth', 'u_lip','l_lip'
# sample_num_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
sample_num_list = [80, 50, 50, 25, 25, 25, 25, 30, 20, 20]
# sample_num_list = [50, 50, 50, 25, 25, 25, 25, 30, 20, 20]
# sample_num_list = [50, 50, 20, 20, 20, 20, 20, 20, 20, 20]
face = cv2.imread(face_path)
cari = cv2.imread(cari_path)
transforms = [torchvision.transforms.Resize(512), ToUnNormalizedTensor()]
transforms = torchvision.transforms.Compose(transforms)


# face_torch = transforms(Image.open(face_path))


def warp_image(image, src_points=None, dst_points=None, transform=None):
    if transform is None:
        if src_points is not None and dst_points is not None:
            transform = get_transform(image, src_points, dst_points)
        else:
            raise Exception('Src points and dst points must not be None.')
    warped = warp(image, transform, output_shape=image.shape)
    return warped, transform


def warp_nearest(image, src_points=None, dst_points=None, transform=None):
    if transform is None:
        if src_points is not None and dst_points is not None:
            transform = get_transform(image, src_points, dst_points)
        else:
            raise Exception('Src points and dst points must not be None.')
    warped = warp(image, transform, output_shape=image.shape, order=0)
    return warped, transform


def get_transform(image, src_points, dst_points):
    src_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + src_points.tolist()
    )
    dst_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + dst_points.tolist()
    )
    tform3 = PiecewiseAffineTransform()
    tform3.estimate(dst_points, src_points)
    return tform3


def sample_arrange(src, num, label):
    """
    Sample key points by equal spaing
    :param src:
    :param num:
    :return:
    """
    arrange = len(src)
    # if num > len(src):
    #     logger.info("Num out of length, return arrange: [{}]".format(src))
    #     return src
    # else:
    # output = np.array((1, 2), dtype=arrange.dtype)
    output = []
    seg = arrange // num
    if seg == 0:
        msg = '[{}]: The number of sampling points exceeds the number of source points, and the original array is ' \
              'equidistantly filled.'.format(label)
        logger.info(msg)
        return insert_equal_space(src, arrange, num)
    seg = arrange / num
    for n in range(num):
        if int(seg * n) >= len(src):
            output.append((src[-1] + src[-2]) // 2)
        else:
            output.append(src[int(seg * n)])
    return output


def insert_equal_space(src, arrange, num):
    output = src.copy()
    need = num - arrange
    sample_space = need // arrange
    mod = need % arrange
    position = 1
    for idx in range(arrange):
        # is_enough = False
        pre_el = src[idx]
        next_el = src[(idx + 1) % arrange]
        output = fill(pre_el, next_el, position, sample_space, output)
        position += (sample_space + 1)
    if len(output) == num:
        return output.reshape(-1, 2)
    else:
        for idx in range(mod):
            output = np.append(output, src[-1])
        return output.reshape(-1, 2)


def fill(pre_el, next_el, position, sample_space, output):
    for j in range(sample_space):
        sample = (pre_el + next_el) // (sample_space + 1) * (j + 1)
        output = np.insert(output, position + j, sample.reshape(2), axis=0)
    return output


def is_filtered(points):
    return len(points) == 1 and (points == np.array([[-1, -1]])).all()


def find_key_points(img, sample_num_list):
    import cv2
    excluded_index = [1, 7]
    labels_tensor = np.arange(0, len(label_list)).reshape(len(label_list), 1, 1)
    # labels_tensor = torch.arange(0, len(label_list)).view(len(label_list), 1, 1)
    split_tensors = (img == labels_tensor).astype(np.uint8)
    point_list_sorted_by_polar = []
    # np.arang
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for index, tensor in enumerate(split_tensors):
        if index in excluded_index:
            # key_points[index] = np.array([[-1, -1]])
            point_list_sorted_by_polar.append(np.array([[-1, -1]]))
            logger.info('Semantic label: [{}] is excluded.'.format(index))
            continue
        color = colormap[tensor].astype(np.uint8)
        label = label_list[index]
        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        # ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        tensor = tensor * 255
        # connects some semantic attribute for generating only one contours
        tensor = cv2.morphologyEx(tensor, cv2.MORPH_CLOSE, kernel)
        ret, binary = cv2.threshold(tensor, 10, 255, cv2.THRESH_BINARY)
        # Skin reverser color ensure finding only on contour
        if index == 0:
            binary = cv2.bitwise_not(binary)
        # if DEBUG:
        #     cv2.imshow('binary', binary)
        #     cv2.waitKey(0)
        tensor, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        logger.info("Semantic label [{}] find contours: [{}]".format(label, len(contours)))
        if not len(contours):
            logger.error('Cannot find contours for semantic label [{}], return None for filtering this img.'.format(
                label))
            return None
            # point_list_sorted_by_polar.append(np.array([[-1, -1]]))
        if len(contours) > 1:
            contours = [max(contours, key=cv2.contourArea)]
        unit_anchor = np.array([0, 1])
        for points in contours:
            mom = cv2.moments(points)
            # print(points.shape)
            centroid = np.array([int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])])
            cv2.circle(color, (centroid[0], centroid[1]), 5, (0, 0, 255), -1)
            points = points.reshape(-1, 2)
            points = [[p, AngleFactory.calAngleClockwise(unit_anchor + centroid, p, centroid)] for p in points]
            points_sorted_by_polar = [el[0] for el in sorted(points, key=lambda el: el[1])]
            logger.info(
                "Semantic label [{}] gains [{}] contour points.".format(label,
                                                                        len(points_sorted_by_polar)))
            point_list_sorted_by_polar.append(points_sorted_by_polar)
            if DEBUG:
                dynamic_display_ordered_contour_points(index, color, points_sorted_by_polar)
    key_point_list = []
    for index, key_points in enumerate(point_list_sorted_by_polar):
        label = label_list[index]
        if is_filtered(key_points):
            logger.info('Semantic tensor [{}] do not contain any contour points or filtered by configuration'.format(
                label))
            key_point_list.append(np.array([[-1, -1]]))
            continue
        sampled_key_point = sample_arrange(key_points, sample_num_list[index], label)
        if len(sampled_key_point) != sample_num_list[index]:
            msg = 'The number of sampling points [{}] must be the same as the number [{}] specified by the configuration in [{}].'.format(
                len(key_points), sample_num_list[index])
            logger.error(msg)
            return None
        logger.debug('Semantic label [{}] sampled: [{}].'.format(label, sampled_key_point))
        key_point_list.append(sampled_key_point)
    return key_point_list
    # centriods.append((center_x, center_y))
    # cv2.circle(color, (center_x, center_y), 4, (152, 255, 255), -1)

    # cv2.imshow('moment', color)
    # cv2.waitKey(0)
    # print(img.shape)
    # print(split_tensors.shape)


def dynamic_display_ordered_contour_points(label_index, color, points_sorted_by_polar):
    tmp_path = 'polar'
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    path = os.path.join(tmp_path, str(label_index))
    if not os.path.exists(path):
        os.mkdir(path)
    # hstack = []
    for index, p in enumerate(points_sorted_by_polar):
        if index % 20 == 0:
            cv2.circle(color, (p[0], p[1]), 4, (152, 255, 255), -1)
            cv2.imwrite(os.path.join(path, str(index)) + '.png', color)
            # hstack.append(color.copy())
    # vstack = []
    # j = 0
    # for index in len(hstack):
    #     if (index + 1) % 4:
    #         vstack.append(np.vstack(hstack[j * 4:index]))
    #         cv2.imwrite(os.path.join(path, str(index)) + '.png', color)
    # cv2.waitKey(0)


def display_pair_key_points(face_src, cari_src, f_kl, c_kl):
    face_img = face_src.copy()
    cari_img = cari_src.copy()
    for index in range(len(f_kl)):
        fpts = f_kl[index]
        cpts = c_kl[index]
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        if is_filtered(fpts) or is_filtered(cpts):
            continue
        for idx in range(len(fpts)):
            cv2.circle(face_img, center=(fpts[idx][0], fpts[idx][1]), radius=2, color=(b, g, r), thickness=-1)
            cv2.circle(cari_img, center=(cpts[idx][0], cpts[idx][1]), radius=2, color=(b, g, r), thickness=-1)
    # cv2.imshow('Key points', img)
    # cv2.waitKey(0)
    return face_img, cari_img


def draw_kpts(src, kpts):
    face_img = src.copy()
    for p in kpts:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        if is_filtered(p):
            continue
        for idx in range(len(p)):
            cv2.circle(face_img, center=(p[0], p[1]), radius=2, color=(b, g, r), thickness=-1)
    # cv2.imshow('Key points', img)
    # cv2.waitKey(0)
    return face_img


def draw_kpts_pil(src, kpts):
    img = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    kpts = kpts.int().numpy().reshape(-1, 2)
    # img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return Image.fromarray(cv2.cvtColor(draw_kpts(img, kpts), cv2.COLOR_BGR2RGB))


def warp_paired(face_img_name, cari_img_name, face_mask_path, cari_mask_path, face_path, cari_path, sample_num_list):
    # test_loader()
    # test_celeb_mask_loading()
    # face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
    # cari_mask = cv2.imread(cari_mask_path, cv2.IMREAD_GRAYSCALE)
    face_color = colormap[face_mask].astype(np.uint8)
    cari_color = colormap[cari_mask].astype(np.uint8)
    face = cv2.imread(face_path)
    cari = cv2.imread(cari_path)
    face = cv2.resize(face, (0, 0), fx=0.5, fy=0.5)
    if face_mask is None:
        logger.info('Loading Img Error, [{}] not found.'.format(face_mask_path))
    # sample_num_list = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    # sample_num_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    ckpts, fkpts, k_cari, k_face = get_paired_key_points(face_img_name, cari_img_name, face_mask, cari_mask,
                                                         sample_num_list, face, cari)
    warped, warped_mask, warped_mask_color, transform = warped_face_mask(ckpts, face, face_color, fkpts)
    # x_position_map, y_position_map = build_position_map(face.shape[1], face.shape[0])
    # x_position_map = make_x_position_map(1,face_mask.shape[1]).reshape()
    # warped_xpm, _ = warp_image(x_position_map, transform=transform)
    # warped_ypm, _ = warp_image(y_position_map, transform=transform)
    # print(x_position_map)
    # delta_x = (warped_xpm * 255).astype(np.uint8) - x_position_map
    # delta_y = (warped_ypm * 255).astype(np.uint8) - y_position_map
    if DEBUG:
        stack = np.hstack((k_face, k_cari, warped))
        stack_mask = np.hstack((face_color, cari_color, warped_mask_color))
        stack_mask = cv2.cvtColor(stack_mask, cv2.COLOR_RGB2BGR)
        stack_all = np.vstack((stack, stack_mask))
        if not os.path.exists(FACE_WARPED):
            os.mkdir(FACE_WARPED)
        cv2.imwrite(os.path.join(FACE_WARPED, str(len(FACE_WARPED) + 1) + '.png'), stack_all)
    return warped_mask


def estimate_offset_field(face_mask, cari_mask, face_img_name, cari_img_name, sample_num_list):
    width, height = face_mask.shape[1], face_mask.shape[0]
    if face_mask is None:
        logger.info('Loading Img Error, [{}] not found.'.format(face_mask_path))
    ckpts, fkpts = get_paired_key_points(face_img_name, cari_img_name, face_mask, cari_mask,
                                         sample_num_list)
    return estimate_offset_field_by_kpts(fkpts, ckpts, height, width)
    # warped_position_map = torch.cat(
    #     [torch.from_numpy(warped_xpm).view(width, width, 1), torch.from_numpy(warped_ypm).view(height, height, 1)],
    #     dim=2).unsqueeze(0)
    # tmp = F.grid_sample(face_torch.unsqueeze(0).double(), warped_position_map)
    # save_image(tmp.long(), 'test.png')
    # print(offset_filed.size())
    # print(delta_x[256:300, 256:300, 0])


def load_warp_from_npy(path):
    if not os.path.exists(path):
        logger.info('Empty path: [{}]'.format(path))
        return None
    else:
        return torch.from_numpy(np.load(path).reshape(1, IMG_SIZE, IMG_SIZE, 2)).float()


def estimate_offset_field_by_kpts(fkpts, ckpts, height, width, x_position_map=None, y_position_map=None):
    delta_x, delta_y = get_split_delta(ckpts, fkpts, height, width, x_position_map, y_position_map)
    offset_filed = torch.cat([delta_x, delta_y], dim=2)
    return offset_filed.unsqueeze(0).float()


def get_split_delta(ckpts, fkpts, height, width, x_position_map=None, y_position_map=None):
    warped_xpm, warped_ypm, x_position_map, y_position_map = get_warped_split_position_map(ckpts, fkpts, height, width,
                                                                                           x_position_map,
                                                                                           y_position_map)
    delta_x = (warped_xpm - x_position_map).view(width, width, 1)
    delta_y = (warped_ypm - y_position_map).view(height, height, 1)
    return delta_x, delta_y


def get_warped_split_position_map(ckpts, fkpts, height, width, x_position_map=None, y_position_map=None):
    if x_position_map is None or y_position_map is None:
        x_position_map, y_position_map = get_split_position_map(height, width)
    warped_xpm, warped_ypm = warp_position_map(ckpts, fkpts, x_position_map, y_position_map)
    x_position_map = torch.from_numpy(x_position_map).float()
    y_position_map = torch.from_numpy(y_position_map).float()
    warped_xpm = torch.from_numpy(warped_xpm).float()
    warped_ypm = torch.from_numpy(warped_ypm).float()
    return warped_xpm, warped_ypm, x_position_map, y_position_map


def get_warped_position_map(ckpts, fkpts, height, width):
    x_position_map, y_position_map = get_split_position_map(height, width)
    warped_xpm, warped_ypm = warp_position_map(ckpts, fkpts, x_position_map, y_position_map)
    # x_position_map = torch.from_numpy(x_position_map)
    # y_position_map = torch.from_numpy(y_position_map)
    warped_xpm = torch.from_numpy(warped_xpm).view(width, width, 1)
    warped_ypm = torch.from_numpy(warped_ypm).view(height, height, 1)
    warped_pm = torch.cat([warped_xpm, warped_ypm], dim=2).unsqueeze(0)
    return warped_pm.float()


def get_split_position_map(height, width):
    x_position_map = make_x_position_map(1, -1, 1, width, height).view(width, width).numpy()
    y_position_map = make_y_position_map(1, -1, 1, width, height).view(height, height).numpy()
    return x_position_map, y_position_map


def warp_position_map(ckpts, fkpts, x_position_map, y_position_map):
    warped_xpm, transform = warp_image(x_position_map, fkpts, ckpts)
    warped_ypm, transform = warp_image(y_position_map, transform=transform)
    return warped_xpm, warped_ypm


def diff_on_same_scale(x, y, start, end):
    return normalize(start, end, x) - (start, end, y)


def normalize(start, end, tensor):
    tensor = tensor.astype(np.float)
    max = np.max(tensor)
    min = np.min(tensor)
    k = (start - end) / (max - min)
    return start + k * (tensor - min)


def warped_face_mask(ckpts, face, face_color, fkpts):
    warped, transform = warp_image(face, fkpts, ckpts)
    # warped, transform = warp_nearest(face, fkpts, ckpts)
    warped = (warped * 255).astype(np.uint8)
    warped_mask, warped_mask_color = warped_color(fkpts, ckpts, face_color, transform)
    return warped, warped_mask, warped_mask_color, transform


def warped_color(fkpts, ckpts, face_color, transform=None):
    warped_mask_color, _ = warp_image(face_color, fkpts, ckpts, transform)
    warped_mask_color = (warped_mask_color * 255).astype(np.uint8)
    warped_mask = image2label(warped_mask_color)
    return warped_mask, warped_mask_color

def warped_color_nearest(fkpts, ckpts, face_color, transform=None):
    warped_mask_color, _ = warp_nearest(face_color, fkpts, ckpts, transform)
    warped_mask_color = (warped_mask_color * 255).astype(np.uint8)
    warped_mask = image2label(warped_mask_color)
    return warped_mask, warped_mask_color

def get_paired_key_points(face_img_name, cari_img_name, face_mask, cari_mask, sample_num_list, face=None, cari=None):
    face_key_point_list = find_key_points(face_mask, sample_num_list)
    cari_key_point_list = find_key_points(cari_mask, sample_num_list)
    # Validate consistency of key points
    fkpts_len = len(face_key_point_list)
    ckpts_len = len(cari_key_point_list)
    if fkpts_len != ckpts_len:
        raise Exception('Face and caricature semantic labels must be consistency.')
    for idx in range(len(face_key_point_list)):
        if len(face_key_point_list[idx]) != len(cari_key_point_list[idx]):
            msg = 'Face [{}] and caricature [{}] key points must be consistency.'.format(face_img_name, cari_img_name)
            raise Exception(msg)
    # merge all attribute key points into one list on warping stage
    fkpts = merge_key_points(face_key_point_list)
    ckpts = merge_key_points(cari_key_point_list)

    merge_key_points(cari_key_point_list, ckpts)
    fkpts = np.array(fkpts)
    ckpts = np.array(ckpts)
    if face is not None and cari is not None:
        k_face, k_cari = display_pair_key_points(face, cari, face_key_point_list, cari_key_point_list)
        return ckpts, fkpts, k_cari, k_face
    return ckpts, fkpts


def merge_key_points(kpts, merged=None):
    merged = []
    for p in kpts:
        merged.extend(p)
    return merged


def make_position_map(batch, start, end, width, height):
    x_position_map = make_x_position_map(batch, start, end, width, height)
    y_position_map = make_y_position_map(batch, start, end, height, height)
    position_map = torch.cat([x_position_map, y_position_map], dim=3)
    return position_map


def make_y_position_map(batch, start, end, width, height):
    height_linspace = torch.linspace(start=start, end=end, steps=height, requires_grad=False, dtype=torch.float).view(
        height, 1)
    y_position_map = height_linspace.expand(width, height).view(1, width, height, 1).expand(batch,
                                                                                            width,
                                                                                            height, 1)
    return y_position_map


def make_x_position_map(batch, start, end, width, height):
    width_linspace = torch.linspace(start=start, end=end, steps=width, requires_grad=False, dtype=torch.float).view(1,
                                                                                                                    width)
    x_position_map = width_linspace.expand(width, height).view(1, width, height, 1).expand(batch,
                                                                                           width,
                                                                                           height, 1)
    return x_position_map


def build_position_map(height, width):
    # face = cv2.imread(face_path)
    # height = face.shape[0]
    # width = face.shape[1]
    x_position_map = np.linspace(start=0, stop=width - 1, num=width, dtype=np.uint8).reshape(1, width)
    x_position_map = np.tile(x_position_map, width - 1, axis=0)
    y_position_map = np.linspace(start=0, stop=height - 1, num=height, dtype=np.uint8).reshape(height, 1)
    y_position_map = np.repeat(y_position_map, height - 1, axis=1)
    print(x_position_map[0, -10:])
    print(y_position_map[0:10, 0])
    return x_position_map, y_position_map


def test_sample_arrange():
    test_pts = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
    pts = sample_arrange(test_pts, 13)
    print(len(pts))
    pts = sample_arrange(test_pts, 14)
    print(len(pts))
    pts = sample_arrange(test_pts, 15)
    print(len(pts))
    pts = sample_arrange(test_pts, 200)
    print(len(pts))


def test_warp_paired():
    msg = 'Expected Key Points sample number for each semantic channel:'
    for index, num in enumerate(sample_num_list):
        msg += ' ,[{}]: [{}]'.format(label_list[index], num)
    logger.info(msg)
    warp_paired(face_img_name, cari_img_name, face_mask_path, cari_mask_path, face_path, cari_path, sample_num_list)


def test_estimate_offset_field():
    offset_field = estimate_offset_field(face_mask, cari_mask, face_img_name, cari_mask_path, sample_num_list)
    print(offset_field.size())


if __name__ == '__main__':
    test_warp_paired()
    # face_color = colormap[face_mask].astype(np.uint8)
    # for i in range(4):
    #     mask = cv2.imread(str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE)
    #     print(np.max(mask))
    #     print(np.min(mask))
    #     face_color = colormap[mask].astype(np.uint8)
    #     cv2.imshow('mask', cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB))
    #     cv2.waitKey(0)
    #
    # colorize()
    # test_estimate_offset_field()
    # test_sample_arrange()
