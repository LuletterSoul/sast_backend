import torch
import os
import yaml
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

matplotlib.use('Agg')


def image_map(img):
    """
    :param img: the image with only one channel opened by PIL.Image
    :return: an image with the format of PTL.Image
    """
    colormap_dict = {0: (0, 0, 0),
                     1: (128, 0, 0),
                     2: (0, 128, 0),
                     3: (128, 128, 0),
                     4: (0, 0, 128),
                     5: (128, 0, 128),
                     6: (0, 128, 128),
                     7: (128, 128, 128),
                     8: (64, 0, 0),
                     9: (192, 0, 0)}
    img_cat = np.vectorize(colormap_dict.get)(img)

    img_cat1 = np.expand_dims(img_cat[0], axis=2)
    img_cat2 = np.expand_dims(img_cat[1], axis=2)
    img_cat3 = np.expand_dims(img_cat[2], axis=2)
    img = np.concatenate([img_cat1, img_cat2, img_cat3], axis=2)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def image_map1(mask):
    colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                         [0, 192, 0], [128, 192, 0], [0, 64, 128]])
    mask = np.array(mask)
    mask_color = colormap[mask].astype(np.uint8)
    mask_color = Image.fromarray(mask_color)
    return mask_color


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [d, m]
      y: pytorch Variable, with shape [d, n]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x = x.t()
    y = y.t()
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def whiten_and_color(cF, sF):
    # cF_size=[c, h, w], sF_size=[c, h, w]
    device = cF.device
    cFSize = cF.size()
    cF = cF.view(cFSize[0], -1)
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] * cFSize[2] - 1)
    _, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    sF = sF.view(sFSize[0], -1)
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] * sFSize[2] - 1)
    _, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    print(
        f'trace={torch.mm((torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t())).t(), (torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t()))).trace()}')
    print(f'norm={torch.norm((torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t()))) ** 2}')

    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature = targetFeature.view(cFSize[0], cFSize[1], cFSize[2])
    return targetFeature


def draw_loss(loss, img_saved_path):
    fig = plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.xlabel('itr')
    plt.ylabel('F')
    plt.title(os.path.splitext(os.path.basename(img_saved_path))[0])
    plt.savefig(img_saved_path)
    plt.close(fig)


def batch_split(feature, patch_size, padding=0, stride=1):
    """
    :param feature: size = [n,c,h,w]
    :param patch_size: (3, 3)
    :param padding: 0
    :param stride: 1
    :return: size = [n, c*kernel_size, L]
    """
    if patch_size == (1, 1):
        n, c, h, w = feature.size()
        feature_unfold = feature.view(n, c, -1)
    else:
        feature_unfold = F.unfold(feature, kernel_size=patch_size, padding=padding, stride=stride)
    # print(f'feature_unfold.size = {feature_unfold.size()}')
    return feature_unfold


def batch_concatenate(feature_unfold, origin_size, patch_size, padding=0, stride=1):
    """
    :param feature_unfold: size = [n, c*kernel_size, L]
    :param origin_size: (h, w)
    :param patch_size: (3, 3)
    :param padding: 0
    :param stride: 1
    :return: size = [n, c, h, w]
    """
    if patch_size == (1, 1):
        n, c, h, w = feature_unfold.size()[0], feature_unfold.size()[1], origin_size[0], origin_size[1]
        feature_fold = feature_unfold.view(n, c, h, w)
    else:
        feature_fold = F.fold(feature_unfold, output_size=origin_size, kernel_size=patch_size, padding=padding,
                              stride=stride)
        ones = torch.ones_like(feature_fold)
        ones_unfold = batch_split(ones, patch_size=patch_size)
        ones_fold = F.fold(ones_unfold, output_size=origin_size, kernel_size=patch_size, padding=padding, stride=stride)
        feature_fold = feature_fold / ones_fold
    return feature_fold


def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']

    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR),
                           dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR),
                         dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(torch.from_numpy(_extract_mask(content_seg, color_codes[i])).unsqueeze(0))
        color_style_masks.append(torch.from_numpy(_extract_mask(style_seg, color_codes[i])).unsqueeze(0))
    color_content_masks = torch.cat(color_content_masks, dim=0)
    color_style_masks = torch.cat(color_style_masks, dim=0)
    return color_content_masks, color_style_masks


def print_options(args):
    args_dict = vars(args)
    option_path = os.path.join(args.output_path, 'options.txt')
    with open(option_path, 'w+') as f:
        print('------------------args---------------------', file=f)
        print('------------------args---------------------')
        for arg_key in args_dict:
            print(f'{arg_key}: {args_dict[arg_key]}', file=f)
            print(f'{arg_key}: {args_dict[arg_key]}')
        print('-------------------end----------------------', file=f)
        print('-------------------end----------------------')


def adjust_learning_rate(optimizer, iteration, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / (1 + iteration * args.lr_decay_rate)


def adjust_tv_loss_weight(args, iteration, ori_tv_loss_weight):
    args.tv_loss_weight = ori_tv_loss_weight / (1 + iteration * args.tv_loss_weight_decay_rate)


def draw_loss(loss_list, content_loss_list, perceptual_loss_list, tv_loss_list, save_path):
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(222)
    plt.plot(range(len(content_loss_list)), content_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('content loss')
    plt.title('content loss')

    plt.subplot(223)
    plt.plot(range(len(perceptual_loss_list)), perceptual_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('perceptual_loss')
    plt.title('perceptual_loss')

    plt.subplot(224)
    plt.plot(range(len(tv_loss_list)), tv_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('tv_loss')
    plt.title('tv_loss')

    # save loss image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def draw_loss_without_style(loss_list, content_loss_list, tv_loss_list, save_path):
    fig = plt.figure()
    plt.subplot(131)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(132)
    plt.plot(range(len(content_loss_list)), content_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('content loss')
    plt.title('content loss')

    plt.subplot(133)
    plt.plot(range(len(tv_loss_list)), tv_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('tv_loss')
    plt.title('tv_loss')

    # save loss image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def draw_loss_(loss_list, content_loss_list, style_loss_list, tv_loss_list, save_path):
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(222)
    plt.plot(range(len(content_loss_list)), content_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('content loss')
    plt.title('content loss')

    plt.subplot(223)
    plt.plot(range(len(style_loss_list)), style_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('style_loss')
    plt.title('style_loss')

    plt.subplot(224)
    plt.plot(range(len(tv_loss_list)), tv_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('tv_loss')
    plt.title('tv_loss')

    # save loss image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def draw_loss_1(loss_list, content_loss_list, edge_loss_list, style_loss_list, tv_loss_list, save_path):
    fig = plt.figure()
    plt.subplot(231)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss')

    plt.subplot(232)
    plt.plot(range(len(content_loss_list)), content_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('content loss')
    plt.title('content loss')

    plt.subplot(233)
    plt.plot(range(len(edge_loss_list)), edge_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('edge_loss')
    plt.title('edge_loss')

    plt.subplot(234)
    plt.plot(range(len(style_loss_list)), style_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('style_loss')
    plt.title('style_loss')

    plt.subplot(235)
    plt.plot(range(len(tv_loss_list)), tv_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('tv_loss')
    plt.title('tv_loss')

    # save loss image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def load_from_yml(args):
    if args.config_path == '':
        print(f'config path is null!')
        return args
    file = open(args.config_path)
    config = yaml.safe_load(file)

    args.description = config['description']
    args.config_path = config['config_path']
    args.content_img = config['content_img']
    args.style_img = config['style_img']
    args.content_dir = config['content_dir']
    args.style_dir = config['style_dir']
    args.segmentation_dir = config['segmentation_dir']
    args.use_seg = config['use_seg']
    args.resize = config['resize']
    args.type = config['type']
    args.encoder_path = config['encoder_path']
    args.decoder_r11_path = config['decoder_r11_path']
    args.decoder_r21_path = config['decoder_r21_path']
    args.decoder_r31_path = config['decoder_r31_path']
    args.decoder_r41_path = config['decoder_r41_path']
    args.decoder_r51_path = config['decoder_r51_path']
    args.layers = config['layers']
    args.is_batch = config['is_batch']
    args.is_combine = config['is_combine']
    args.is_select = config['is_select']
    args.select_content_list = config['select_content_list']
    args.select_style_list = config['select_style_list']
    args.orth_constraint = config['orth_constraint']
    args.post_smoothing = config['post_smoothing']
    args.fast = config['fast']
    args.output_path = config['output_path']
    args.gpu = config['gpu']
    args.device = config['device']
    args.max_use_num = config['max_use_num']
    args.soft_lambda = config['soft_lambda']
    args.k_cross = config['k_cross']
    args.patch_size = config['patch_size']
    args.style_weight = config['style_weight']
    args.reduce_dim_type = config['reduce_dim_type']
    args.dist_type = config['dist_type']
    args.dim_thresh = config['dim_thresh']
    args.skip_connection = config['skip_connection']
    args.connect_weight = config['connect_weight']
    args.skip_connection_type = config['skip_connection_type']
    args.skip_connection_decoder_path = config['skip_connection_decoder_path']


def main():
    # size = [1, 512, 64, 64]
    # patch_size = (1, 1)
    # cf = torch.rand(size=size)
    # a = batch_split(cf, patch_size=patch_size)
    # res = batch_concatenate(a, origin_size=(64, 64), patch_size=patch_size)
    # print(torch.sum(cf - res))

    # a = torch.tensor([[[1, 2], [3, 4]],
    #                   [[5, 6], [7, 8]]])
    # b = a.reshape(8, 1)

    # test adjust_tv_loss_weight()
    ori_tv_weight = 1e-6
    for iteration in range(1, 19572 * 4 + 1):
        print(f'iteration={iteration}, weight={ori_tv_weight / (1 + iteration * 1e-3)}')


if __name__ == '__main__':
    main()

    # plt.figure(1)
    # plt.subplot(221)
    # plt.scatter([1, 3, 5], [2, 4, 6])
    # plt.title('221')
    # plt.xlabel('x1')
    # plt.ylabel('y1')
    #
    # plt.subplot(222)
    # plt.plot([1, 3, 5], [2, 4, 6])
    # plt.title('222')
    # plt.xlabel('x2')
    # plt.ylabel('y2')
    #
    # plt.subplot(223)
    # plt.plot([1, 3, 5], [2, 4, 6])
    # plt.title('223')
    # plt.xlabel('x3')
    # plt.ylabel('y3')
    #
    # plt.subplot(224)
    # plt.scatter([1, 3, 5], [2, 4, 6])
    # plt.title('224')
    # plt.xlabel('x4')
    # plt.ylabel('y4')
    #
    # plt.tight_layout()
    # plt.savefig('res.png')
