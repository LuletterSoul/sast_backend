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
from .warp import *
import torch


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
