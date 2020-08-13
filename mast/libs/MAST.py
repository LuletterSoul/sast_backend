# coding=UTF-8
import torch
import gc
import os
import matplotlib
import numpy as np
import cv2 as cv
import torch.nn.functional as f
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .utils import batch_split, batch_concatenate, load_seg, euclidean_dist
from .PatchMatch import patch_match_split, soft_patch_match_split

matplotlib.use('Agg')


class MAST(object):
    def __init__(self, args):
        """
        the type of all values are the same as cf
        :param args: parser args for command
        """
        self.args = args
        if args.reduce_dim_type == 'avg_pool':
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif args.reduce_dim_type == 'max_pool':
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def cal_p(self, cf, sf, mask=None):
        """
        :param cf: [c*kernel_size, hcwc]
        :param sf: [c*kernel_size, hsws]
        :param mask: [hcwc, hsws]
        :return:p [c*c]
        """
        cf_size = cf.size()
        sf_size = sf.size()
        k_cross = self.args.k_cross

        cf_temp = cf.clone()
        sf_temp = sf.clone()

        if self.args.max_use_num == -1:
            # ########################################
            # normalize
            cf_n = f.normalize(cf, 2, 0)
            sf_n = f.normalize(sf, 2, 0)
            # #########################################

            if self.args.dist_type == 'cosine':
                dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar
            elif self.args.dist_type == 'euclidean':
                dist = euclidean_dist(cf, sf)
            if mask is not None:
                mask = mask.type_as(dist).to(self.args.device)
                dist = torch.mul(dist, mask)

            hcwc, hsws = cf_size[1], sf_size[1]
            U = torch.zeros(hcwc, hsws).type_as(cf_n).to(self.args.device)  # construct affinity matrix "(h*w)*(h*w)"

            index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
            value = torch.ones(k_cross, hsws).type_as(cf_n).to(self.args.device)  # "KCross*(h*w)"
            U.scatter_(0, index, value)  # set weight matrix
            del index
            del value
            gc.collect()

            index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
            value = torch.ones(hcwc, k_cross).type_as(cf_n).to(self.args.device)
            U.scatter_(1, index, value)  # set weight matrix
            del index
            del value
            gc.collect()
        elif self.args.max_use_num == 0:
            U = soft_patch_match_split(cf, sf, soft_lambda=self.args.soft_lambda)
        else:
            U = patch_match_split(cf, sf, max_use_num=self.args.max_use_num)
            # U = patch_match(cf=self.cf, sf=self.sf, max_use_num=self.args.max_use_num)
        n_cs = torch.sum(U)
        U = U / n_cs
        D1 = torch.diag(torch.sum(U, dim=1)).type_as(cf).to(self.args.device)
        if self.args.orth_constraint:
            A = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
            regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.args.device) * 1e-12
            # A += regularization_term
            A_U, A_S, A_V = torch.svd(A)
            p = torch.mm(A_U, A_V.t())
        else:
            try:
                A = torch.mm(torch.mm(cf_temp, D1), cf_temp.t())
                regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.args.device) * 1e-12
                A += regularization_term
                B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
                p = torch.solve(B, A).solution

                # A = torch.mm(torch.mm(cf_temp, Dc), cf_temp.t())
                # A_inv = torch.pinverse(A)
                # B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
                # p = torch.mm(A_inv, B)
            except Exception as e:
                print(e)
                p = torch.eye(cf_size[0]).type_as(cf).to(self.args.device)
        return p

    def cal_csf(self, ori_cf, cf, sf, mask=None):
        """
        :param ori_cf:
        :param cf: [n, c*kernel_size, hcwc]
        :param sf: [n, c*kernel_size, hsws]
        :param mask: [hcwc, hsws]
        :return: csf [n, c*kernel_size, hcwc]
        """
        cf_size = cf.size()
        sf_size = sf.size()
        if cf_size[0] != sf_size[0] or cf_size[1] != sf_size[1]:
            csf = cf
        else:
            csf = []
            for i in range(cf_size[0]):
                ori_cf_temp = ori_cf[i]
                cf_temp = cf[i]
                sf_temp = sf[i]
                p = self.cal_p(cf_temp, sf_temp, mask)
                csf_temp = torch.mm(p.t(), ori_cf_temp).unsqueeze(0)
                csf.append(csf_temp)
            csf = torch.cat(csf, dim=0)
        return csf

    def can_seg(self, content_seg_path, style_seg_path):
        if self.args.patch_size != 1:
            print(f'patch size = {self.args.patch_size}, must be 1, can not use segmentation...')
            return False
        if self.args.max_use_num != -1:
            print(f'max use num={self.args.max_use_num}, must be -1, can not use segmentation...')
            return False
        if not os.path.exists(content_seg_path):
            print(f'content segmentation image [{content_seg_path}] not exists...')
            return False
        if not os.path.exists(style_seg_path):
            print(f'style segmentation image [{style_seg_path}] not exists...')
            return False
        return True

    def down_sampling_feature(self, cf, sf):
        thresh = self.args.dim_thresh * self.args.dim_thresh
        while cf.size()[2] * cf.size()[3] > thresh:
            cf = self.pool(cf)
        while sf.size()[2] * sf.size()[3] > thresh:
            sf = self.pool(sf)
        return cf, sf

    def cal_mask(self, c_mask, s_mask, cf_size, sf_size):
        hc, wc = cf_size[2], cf_size[3]
        hs, ws = sf_size[2], sf_size[3]
        c_mask = cv.resize(c_mask, (hc, wc), cv.INTER_NEAREST)
        s_mask = cv.resize(s_mask, (hs, ws), cv.INTER_NEAREST)
        max_color_index = min(np.max(c_mask), np.max(s_mask))
        c_mask_tensor = torch.from_numpy(c_mask)
        s_mask_tensor = torch.from_numpy(s_mask)
        mask = torch.zeros(hc * wc, hs * ws).int()
        for color_index in range(0, max_color_index + 1):
            c_pos = torch.where(c_mask_tensor == color_index)
            s_pos = torch.where(s_mask_tensor == color_index)
            c_index = c_pos[0] * hc + c_pos[1]
            s_index = s_pos[0] * hs + s_pos[1]
            col = torch.zeros(hc * wc, 1).int()
            col.index_fill_(0, c_index, 1)
            row = torch.zeros(1, hs * ws).int()
            row.index_fill_(1, s_index, 1)
            temp = torch.mm(col, row)
            mask = mask.__or__(temp)
        return mask

    def transform(self, cf, sf, c_mask, s_mask):
        """
        :param cf: [n, c, hc, wc]
        :param sf: [n, c, hs, ws]
        :param c_mask: content mask
        :param s_mask: style mask
        :return: csf [n, c, hc, wc]
        """
        ori_cf = cf.clone()
        ori_sf = sf.clone()
        ori_cf_size = ori_cf.size()
        ori_sf_size = ori_sf.size()

        cf, sf = self.down_sampling_feature(cf, sf)
        cf_size = cf.size()
        sf_size = sf.size()
        # print(f'ori_cf_size={ori_cf_size}, cf_size={cf_size}, ori_sf_size={ori_sf_size}, sf_size={sf_size}')
        if c_mask is not None and s_mask is not None:
            mask = self.cal_mask(c_mask, s_mask, cf_size, sf_size)
        else:
            mask = None

        cf_split = batch_split(cf, patch_size=(self.args.patch_size, self.args.patch_size))
        sf_split = batch_split(sf, patch_size=(self.args.patch_size, self.args.patch_size))
        ori_cf_split = batch_split(ori_cf, patch_size=(self.args.patch_size, self.args.patch_size))
        csf = self.cal_csf(ori_cf_split, cf_split, sf_split, mask)
        csf = batch_concatenate(csf, origin_size=(ori_cf_size[2], ori_cf_size[3]),
                                patch_size=(self.args.patch_size, self.args.patch_size))
        return csf


if __name__ == '__main__':
    pass
