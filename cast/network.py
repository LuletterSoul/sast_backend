import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn as nn, nn
from torch.nn import functional as F
from torch.utils import data as data


# from NeuralStyleTransfer import style_feats, content_feats, laplacian_s_feats, style_targets, content_targets, \
#     laplacia_targets, targets, loss_fns


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


class ConsistencyLoss(nn.Module):

    def __init__(self, laplacian_graph, k, mask=None, mean='mean') -> None:
        super().__init__()
        self.mask = mask
        self.consistency_matrix = None
        self.laplacian_graph = laplacian_graph
        self.k = k
        self.mean = mean

    def forward(self, input, target):
        # size = input.size()[-2:]
        N, C, H, W = input.size()
        # print(size)
        #
        # if self.consistency_matrix == None and mask is not None:
        #     down_sampled_mask = F.interpolate(mask, size)
        #     flat_mask_col = down_sampled_mask.view(-1, 1)
        #     flat_mask_row = down_sampled_mask.view(1, -1)
        #     self.consistency_matrix = (flat_mask_col == flat_mask_row).float()
        # if self.consistency_matrix == None:
        flat_input = input.view(C, -1)
        flat_target = target.view(C, -1)
        # print(flat_target.size())
        # print(flat_input.size())
        dist_matrix = cal_dist(flat_input, flat_target)
        assert self.laplacian_graph.size() == dist_matrix.size()
        weighted_dist_matrix = self.laplacian_graph * dist_matrix
        # print(
        #     f'Dist matrix total features: {weighted_dist_matrix.size()}')
        # return weighted_dist_matrix.mean()
        if self.mean == 'mean':
            return weighted_dist_matrix.sum() / (H * W * self.k)
        else:
            return weighted_dist_matrix.mean()


def cal_dist(A, B):
    """
    :param A: (d, m) m个d维向量
    :param B: (d, n) n个d维向量
    :return: (m, n)
    """
    ASize = A.size()
    BSize = B.size()
    dist = (torch.sum(A ** 2, dim=0).view(ASize[1], 1).expand(ASize[1], BSize[1]) +
            torch.sum(B ** 2, dim=0).view(1, BSize[1]).expand(ASize[1], BSize[1]) -
            2 * torch.mm(A.t(), B)).to(A.device)
    return dist


def cal_graph(content_feat, style_feat, device, k=3, reverse=False, c_mask=None, s_mask=None, use_mask=False):
    print(f'Content feature map size: {content_feat.size()}')
    print(f'Style feature map size: {style_feat.size()}')
    assert content_feat.size() == style_feat.size()
    N, C, H, W = content_feat.size()
    content_feat = content_feat.squeeze()
    style_feat = style_feat.squeeze()
    n_content_feat = F.normalize(content_feat, dim=0).view(C, -1)
    n_style_feat = F.normalize(style_feat, dim=0).view(C, -1)
    cosine_dist_matrix = torch.mm(n_content_feat.t(), n_style_feat)
    mask = torch.ones(H * W, H * W, dtype=n_content_feat.dtype).to(device)
    if reverse:
        cosine_dist_matrix *= -1

    # print('Cosin dis matrix size: [{}]'.format(cosince_dist_matrix.size()))
    if c_mask is not None and s_mask is not None and use_mask:
        print('Use mask guide.')
        c_mask = F.interpolate(c_mask.float(), [H, W]).long().view(H * W, 1)
        s_mask = F.interpolate(s_mask.float(), [H, W]).long().view(1, H * W)
        # test = (c_mask == s_mask).long()
        # print(torch.sum(test))
        mask = (c_mask == s_mask).type_as(n_content_feat)

    cosine_dist_matrix *= mask
    graph = torch.zeros(H * W, H * W, dtype=n_content_feat.dtype).to(device)
    index = torch.topk(cosine_dist_matrix, k, 0)[1]
    value = torch.ones(k, H * W, dtype=n_content_feat.dtype).to(device)
    graph.scatter_(0, index, value)  # set weight matrix

    del index
    del value

    index = torch.topk(cosine_dist_matrix, k, 1)[1]
    value = torch.ones(H * W, k, dtype=n_content_feat.dtype).to(device)
    graph.scatter_(1, index, value)  # set weight matrix

    del index
    del value
    del cosine_dist_matrix

    return graph


def build_label_map():
    labels = torch.arange(0, 10).view(1, 10, 1, 1).expand(1,
                                                          10, 1,
                                                          1).long()
    return labels


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, mask_root, tf1, tf2, mask_tf):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        # self.mask_paths = list(Path(self.mask_paths).glob())
        # self.transform = transform
        self.mask_root = mask_root
        self.tf1 = tf1
        self.tf2 = tf2
        self.mask_tf = mask_tf

    def __getitem__(self, index):
        path = str(self.paths[index])
        idx = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert('RGB')
        mask_path = os.path.join(self.mask_root, idx + '.png')
        if not os.path.exists(mask_path):
            mask = 0
        else:
            mask = Image.open(mask_path)
            mask = self.mask_tf(mask)
        img_low = self.tf1(img)
        img_high = self.tf2(img)

        return img_low, img_high, mask, idx

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


class Maintainer:

    def __init__(self, vgg, content_image, style_image, content_layers, style_layers,
                 laplacian_layers, device, kl=7, km=1, c_mask=None, s_mask=None, use_mask=False, mean='mean') -> None:
        self.vgg = vgg
        self.content_image = content_image
        self.style_image = style_image
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.laplacian_layers = laplacian_layers
        self.mutex_flag = False
        self.c_mask = c_mask
        self.s_mask = s_mask
        self.kl = kl
        self.km = km
        self.device = device
        self.laplacian_updated = False
        self.use_mask = use_mask
        self.mean = mean

        self.build_training_components(content_image, style_image, content_layers, style_layers, laplacian_layers,
                                       c_mask, s_mask)

    def build_layers(self, img, keys):
        # global laplacia_c_feats, laplacia_s_feats
        # laplacia_c_feats = vgg(content_image_hr, laplacia_layers)
        # laplacia_s_feats = vgg(style_image_hr, laplacia_layers)
        return self.vgg(img, keys)

    def build_loss_fns(self):
        loss_fns = [GramMSELoss()] * len(self.style_layers) + [nn.MSELoss()] * len(self.content_layers) + [
            ConsistencyLoss(l, self.kl, mean=self.mean) for
            l in self.laplacian_graph]
        if torch.cuda.is_available():
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        return loss_fns

    # def build_laplacian_graph(self, c_feats, s_feats):
    #     laplacian_graphs = [
    #         cal_graph(c_feats[idx], s_feats[idx], self.device,
    #                   7) for idx in range(len(c_feats))]
    #     return laplacian_graphs

    def build_graph(self, c_feats, s_feats, k, reverse=False, c_mask=None, s_mask=None):
        """
        cal neigborhood  graph
        :param c_feats:
        :param s_feats:
        :param reverse: cal Top-k most similar when  false， cal Top-k most not-similar when true
        :return:
        """
        assert len(c_feats) == len(s_feats)
        graph = [
            cal_graph(c_feats[idx], s_feats[idx], self.device,
                      k, reverse, c_mask, s_mask, self.use_mask) for idx in range(len(c_feats))]

        return graph

    def build_training_components(self, content_image, style_image, content_layers, style_layers,
                                  laplacian_layers_keys, c_mask=None, s_mask=None):
        self.style_feats = self.build_layers(style_image, style_layers)
        self.content_feats = self.build_layers(content_image, content_layers)
        self.laplacian_c_feats = self.build_layers(content_image, laplacian_layers_keys)
        self.laplacian_s_feats = self.build_layers(style_image, laplacian_layers_keys)

        self.style_targets = [GramMatrix()(A).detach() for A in self.style_feats]
        self.content_targets = [A.detach() for A in self.content_feats]
        self.laplacia_targets = [A.detach() for A in self.laplacian_s_feats]
        self.targets = self.style_targets + self.content_targets + self.laplacia_targets

        self.laplacian_graph = self.build_graph(self.laplacian_c_feats, self.laplacian_s_feats, self.kl, reverse=False,
                                                c_mask=c_mask, s_mask=s_mask)
        self.loss_fns = self.build_loss_fns()

    def add_mutex_constrain(self, feats):
        # self.mutex_feats = self.build_layers(img, layers)
        mutex_targets = [A.detach() for A in feats]
        torch.cuda.empty_cache()
        if not self.mutex_flag:
            mutex_graph = self.build_graph(feats, feats, self.km, True)
            self.targets += mutex_targets
            self.loss_fns += [ConsistencyLoss(l) for
                              l in mutex_graph]
            self.mutex_flag = True
        else:
            mutex_graph = self.build_graph(feats, feats, self.km, True)
            index = len(self.style_feats) + len(self.content_targets) + len(self.laplacia_targets)
            # del self.targets[-len(mutex_targets):]
            self.targets[:index] += mutex_targets
            self.loss_fns[:index] += [ConsistencyLoss(l) for
                                      l in mutex_graph]

    def update_loss_fns_with_lg(self, c_feats, s_feats, k):
        for l in self.laplacian_graph:
            del l
        c_feats = [A.detach() for A in c_feats]
        s_feats = [A.detach() for A in s_feats]
        self.laplacian_graph = self.build_graph(c_feats, s_feats, k)
        self.loss_fns = self.build_loss_fns()
        torch.cuda.empty_cache()
