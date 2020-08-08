import torch
import time
import numpy as np
import random
import torch.nn.functional as f


def patch_match(cf: torch.Tensor, sf: torch.Tensor, max_use_num):
    """

    :param cf: (C, Hc, Wc)
    :param sf: (C, Hs, Ws)
    :param max_use_num: max_use_num for each pixel
    :return: f (HcWc, HsWs)
    """
    device = cf.device
    cf_size = cf.size()
    sf_size = sf.size()
    cf_n = f.normalize(cf, 2, 0).view(cf_size[0], -1)  # (C, HcWc)
    sf_n = f.normalize(sf, 2, 0).view(sf_size[0], -1)  # (C, HsWs)
    residue_use_num = torch.ones(sf_size[1] * sf_size[2]).type(torch.int).to(device) * max_use_num
    res = torch.zeros(cf_size[1] * cf_size[2], sf_size[1] * sf_size[2]).type_as(cf).to(device)
    sample_list = random.sample(range(cf_size[1] * cf_size[2]), cf_size[1] * cf_size[2])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        residue_use_num[max_pos] -= 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
        if residue_use_num[max_pos] == 0:
            sf_n[:, max_pos] = 0
    return res


def patch_match_split(cf_split: torch.Tensor, sf_split: torch.Tensor, max_use_num):
    """

    :param cf_split: (c*kernel_size, L)
    :param sf_split: (c*kernel_size, L)
    :param max_use_num: max_use_num for each pixel
    :return: f (HcWc, HsWs)
    """
    device = cf_split.device
    cf_size = cf_split.size()
    sf_size = sf_split.size()
    cf_n = f.normalize(cf_split, 2, 0)  # (c*kernel_size, L)
    sf_n = f.normalize(sf_split, 2, 0)  # (c*kernel_size, L)
    residue_use_num = torch.ones(sf_size[1]).type(torch.int).to(device) * max_use_num
    res = torch.zeros(cf_size[1], sf_size[1]).type_as(cf_split).to(device)
    sample_list = random.sample(range(cf_size[1]), cf_size[1])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        residue_use_num[max_pos] -= 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
        if residue_use_num[max_pos] == 0:
            sf_n[:, max_pos] = 0
    return res


def soft_patch_match_split(cf_split: torch.Tensor, sf_split: torch.Tensor, soft_lambda):
    """

    :param cf_split: (c*kernel_size, L)
    :param sf_split: (c*kernel_size, L)
    :param soft_lambda:
    :return: f (HcWc, HsWs)
    """
    device = cf_split.device
    cf_size = cf_split.size()
    sf_size = sf_split.size()
    cf_n = f.normalize(cf_split, 2, 0)  # (c*kernel_size, L)
    sf_n = f.normalize(sf_split, 2, 0)  # (c*kernel_size, L)
    use_num = torch.zeros(1, sf_size[1]).type_as(cf_split).to(device)
    res = torch.zeros(cf_size[1], sf_size[1]).type_as(cf_split).to(device)
    sample_list = random.sample(range(cf_size[1]), cf_size[1])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        dist -= soft_lambda * use_num
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        use_num[0][max_pos] += 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
    return res


def main():
    # f_size = (512, 64, 64)
    f_size = (512 * 9, 64 * 64)
    cf = torch.rand(size=f_size)
    sf = torch.rand(size=f_size)
    max_use_num = 2
    soft_lambda = 0.05
    start_time = time.time()
    # res = patch_match_split(cf, sf, max_use_num)
    res = soft_patch_match_split(cf, sf, soft_lambda)
    print(res.size())
    print(f'used time={time.time() - start_time}')
    print(torch.sum(torch.sum(res, dim=1)))
    print(torch.tensor(np.nan))


if __name__ == '__main__':
    main()
