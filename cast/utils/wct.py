#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: wct.py
@time: 10/24/19 3:13 PM
@version 1.0
@desc:
"""
# from datasets.g_anno import *
from utils.warp import *
from utils.misc import *

# photo dataset
#  

def cal_tensor_mean(tensor):
    c_mean = torch.mean(tensor, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(tensor)
    return c_mean


def whiten_and_color_mu(cF, sF, test=None):
    cFSize = cF.size()
    # c_mean = cal_tensor_mean(cF)
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    face_mu = c_mean.view(-1, 2).long().cpu().numpy()
    # c_mean_expand = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean.unsqueeze(1).expand_as(cF)

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(cF.device)
    c_u, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF, 1)
    cari_mu = s_mean.view(-1, 2).long().cpu().numpy()
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
    s_u, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    if sFSize[1] == 1:
        return
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))

    s_d = (s_e[0:k_s]).pow(0.5)

    whiten_cF = torch.mm(step2, cF)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)

    if test is not None:
        test = test - c_mean.unsqueeze(1).expand_as(test)
        whiten_test = torch.mm(step2, test)
        testFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_test)
        testFeature = testFeature + s_mean.unsqueeze(1).expand_as(testFeature)
        return targetFeature, testFeature, face_mu, cari_mu
    return targetFeature, face_mu, cari_mu


def whiten_and_color(cF, sF, test=None):
    cFSize = cF.size()
    # c_mean = cal_tensor_mean(cF)
    c_mean = torch.mean(cF, 1)  # c x (h x w)

    # c_mean_expand = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean.unsqueeze(1).expand_as(cF)

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(cF.device)
    c_u, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
    if sFSize[1] > 1:
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        s_d = (s_e[0:k_s]).pow(0.5)
        whiten_cF = torch.mm(step2, cF)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    else:
        targetFeature = s_mean.unsqueeze(1).expand_as(cF)
        testFeature = s_mean.unsqueeze(1).expand_as(test)
        return targetFeature, testFeature
    if test is not None:
        test = test - c_mean.unsqueeze(1).expand_as(test)
        whiten_test = torch.mm(step2, test)
        testFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_test)
        testFeature = testFeature + s_mean.unsqueeze(1).expand_as(testFeature)
        return targetFeature, testFeature
    return targetFeature


def whiten_and_color(cF, sF, test=None, c_mean=None, s_mean=None, use_mean=False):
    cFSize = cF.size()
    # c_mean = cal_tensor_mean(cF)
    if c_mean is None:
        c_mean = torch.mean(cF, 1)  # c x (h x w)

    # c_mean_expand = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean.unsqueeze(1).expand_as(cF)

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1]) + torch.eye(cFSize[0]).double().to(cF.device)
    c_u, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    if s_mean is None:
        s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1])
    if sFSize[1] > 1 or not use_mean:
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        s_d = (s_e[0:k_s]).pow(0.5)
        whiten_cF = torch.mm(step2, cF)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        if test is not None:
            test = test - c_mean.unsqueeze(1).expand_as(test)
            whiten_test = torch.mm(step2, test)
            testFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_test)
            testFeature = testFeature + s_mean.unsqueeze(1).expand_as(testFeature)
            return targetFeature, testFeature
    elif use_mean:
        targetFeature = s_mean.unsqueeze(1).expand_as(cF)
        testFeature = s_mean.unsqueeze(1).expand_as(test)
        return targetFeature, testFeature
    return targetFeature


def test_wct():
    offset = 500

    face_landmarks = []
    cari_landmarks = []

    face_landmarks_names = get_filenames(FACE_LANDMARKS_PATH, offset)
    cari_landmarks_names = get_filenames(CARI_LANDMARKS_PATH)

    print(face_landmarks_names)

    for face_landmark_name in face_landmarks_names:
        face_landmarks.append(np.loadtxt(os.path.join(FACE_LANDMARKS_PATH, face_landmark_name)).reshape(-1, 1))

    for cari_landmark_name in cari_landmarks_names:
        cari_landmarks.append(np.loadtxt(os.path.join(CARI_LANDMARKS_PATH, cari_landmark_name)).reshape(-1, 1))

    face_landmarks = np.array(face_landmarks, dtype=np.int).reshape((offset, 272, 2))
    cari_landmarks = np.array(cari_landmarks, dtype=np.int).reshape((len(cari_landmarks_names), 272, 2))

    print(face_landmarks.shape)

    fl_tensor = torch.from_numpy(face_landmarks).view(offset, -1).permute(1, 0).double()
    cl_tensor = torch.from_numpy(cari_landmarks).view(len(cari_landmarks_names), -1).permute(1, 0).double()

    print(fl_tensor.size())
    print(cl_tensor.size())

    transfer = whiten_and_color(fl_tensor, cl_tensor)

    transfer = transfer.permute(1, 0).view(offset, 272, -1).long()
    print(transfer.size())

    # face_names = get_filenames(FACE_IMG_PATH, offset)

    save_dir = os.path.join(FACE_PATH, 'wct')
    compare_dir = os.path.join(FACE_PATH, 'cmp')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(compare_dir):
        os.mkdir(compare_dir)

    for idx, face_name in enumerate(face_landmarks_names):
        # offset_filed = estimate_offset_field_by_kpts(face_landmarks[idx], transfer[idx], IMG_SIZE, IMG_SIZE,
        #                                              x_position_map,
        #                                              y_position_map)
        face_name = os.path.splitext(face_name)[0] + '.jpg'
        print('Processing: [{}]'.format(face_name))
        face = cv2.resize(cv2.imread(os.path.join(FACE_IMG_PATH, face_name)), (0, 0),
                          fx=0.5, fy=0.5)
        warped, _ = warp_image(face, face_landmarks[idx], transfer[idx])
        warped = (warped * 255).astype(np.uint8)
        warped_kpts = draw_key_points(warped, transfer[idx])
        face = draw_key_points(face, face_landmarks[idx])
        cmp = np.hstack((face, warped_kpts))
        cv2.imwrite(os.path.join(save_dir, face_name), warped)
        cv2.imwrite(os.path.join(compare_dir, face_name), cmp)
        print('Processed: [{}]'.format(face_name))


if __name__ == '__main__':
    test_wct()
