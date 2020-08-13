#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: g_anno.py
@time: 10/2/19 3:00 PM
@version 1.0
@desc: generate names list for each dataset
"""
import multiprocessing as mp
import pickle

from utils.misc import *
from utils.warp import *
from utils.wct import whiten_and_color
from constant import *

# FACE_DATASET_PATH = './CelebAMaskHQ-mask'
# FACE_LANDMARKS_PATH = './CelebAMaskHQ-mask-landmarks'
# CARI_DATASET_PATH = './Caricature-mask'
# CARI_LANDMARKS_PATH = './Caricature-mask-landmarks'

# 'skin', 'nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'mouth', 'u_lip','l_lip'
# sample_num_list = [50, 50, 30, 30, 30, 30, 30, 30, 30, 30]
x_position_map, y_position_map = get_split_position_map(IMG_SIZE, IMG_SIZE)


def generate_name_list(current_dir, write=True, offset=None):
    dirs = get_dirs(current_dir)
    if len(dirs) == 0:
        print('Empty dirs')
        return
    d_pair = []
    filename_list = []
    for d, jd in dirs:
        childs = sorted_names(jd)
        if len(childs) == 0:
            print('Empty childs')
            return
        filename_list.append([os.path.splitext(c)[0] for c in childs])
        d_pair.append((d, os.path.splitext(childs[0])[1]))
        # d_pair.append((d, childs))
        # file = open(d + '.txt', 'w')
        # if write:
        #     with open(d + '.txt', 'w') as f:
        #         for item in childs:
        #             f.write("%s\n" % item)

    intersection = filename_list[0]
    for f in filename_list:
        intersection = [c for c in f if c in intersection]

    if offset is not None:
        intersection = intersection[:offset]

    for d, extention in d_pair:
        print('Processing [{}]....'.format(d))
        with open(os.path.join(current_dir, d + '.txt'), 'w') as f:
            for item in intersection:
                f.write("%s\n" % (item + extention))
        print('Processed done [{}]....'.format(d))


def is_img_format(extention):
    return extention == '.jpg' or extention == '.png'


def make_landmarks(path, save_path, offset=None):
    # file_names = sorted_names(path)
    file_names = get_filenames(path, offset)
    if len(file_names) <= 0:
        msg = 'No file exists in [{}].'.format(path)
        raise Exception(msg)
    if not os.path.exists(save_path):
        print('Make new dir: [{}]'.format(save_path))
        os.mkdir(save_path)
    # process_pool = mp.Pool(processes=mp.cpu_count() - 1)
    args = [(path, name, save_path) for name in file_names]
    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(landmarks_task, args)

    #     t.map(landmarks_task, args)
    # for name in file_names:
    #     process_pool.apply_async(landmarks_task, args=(path, name, save_path, sample_num_list))
    # process_pool.close()
    # process_pool.join()


def landmarks_task(path, name, save_path):
    filename, extention = os.path.splitext(name)
    if not is_img_format(extention):
        print('Filtered file [{}].'.format(name))
        return False
    print('Processing [{}]...........'.format(name))
    img_path = os.path.join(path, name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Loading img [{}] Error, emtpy img.'.format(name))
        return False
    pts = find_key_points(img, sample_num_list)
    if pts is None:
        return False
    key_points = merge_key_points(pts)
    np.savetxt(os.path.join(save_path, filename + '.txt'), key_points, fmt='%d')
    print('Processed [{}].........'.format(name))
    return True


def make_wct_landmarks(face_path: DatasetPath, cari_path: DatasetPath,
                       offset, face_test, device_id,
                       landmarks_num=272, enable_draw=False, face_scale=0.5, cari_scale=2):
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'
    face_landmarks = []
    test_face_landmarks = []
    cari_landmarks = []

    total_landmarks_names = get_filenames(face_path.landmarks, offset)
    face_landmarks_names = total_landmarks_names[0:face_test]
    test_face_landmarks_names = total_landmarks_names[face_test:]
    cari_landmarks_names = get_filenames(cari_path.landmarks)

    # print(face_landmarks_names)

    print('Loading face landmarks from [{}]'.format(face_path.landmarks))
    for face_landmark_name in face_landmarks_names:
        print('Loading face [{}]...'.format(face_landmark_name))
        face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)))

    for test_face_landmark_name in test_face_landmarks_names:
        # face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)).reshape(-1, 1))
        print('Loading face [{}]...'.format(test_face_landmark_name))
        test_face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, test_face_landmark_name)))

    print('Loading face landmarks done.')

    print('Loading cari landmarks from [{}]'.format(cari_path.landmarks))
    for cari_landmark_name in cari_landmarks_names:
        # cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)).reshape(-1, 1))
        print('Loading cari [{}]...'.format(cari_landmark_name))
        cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)))
    print('Loading cari landmarks done.')

    # face_landmarks = np.array(face_landmarks, dtype=np.int).reshape((offset, landmarks_num, 2))
    # cari_landmarks = np.array(cari_landmarks, dtype=np.int).reshape((len(cari_landmarks_names), landmarks_num, 2))
    face_landmarks = (np.array(face_landmarks) * face_scale).astype(np.int)
    test_face_landmarks = (np.array(test_face_landmarks) * face_scale).astype(np.int)
    cari_landmarks = (np.array(cari_landmarks) * cari_scale).astype(np.int)

    fl_tensor = torch.from_numpy(face_landmarks).view(offset - face_test, -1).permute(1, 0).double().to(device)
    test_fl_tensor = torch.from_numpy(test_face_landmarks).view(face_test, -1).permute(1, 0).double().to(device)
    cl_tensor = torch.from_numpy(cari_landmarks).view(len(cari_landmarks_names), -1).permute(1, 0).double().to(device)

    # print(fl_tensor.size())
    # print(cl_tensor.size())

    start = time.time()
    print('Processing whiting and color operation: Total content features:[{}]/ style feature: [{}]'.format(
        len(face_landmarks_names), len(cari_landmarks_names)))
    wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    face_landmarks = face_landmarks.reshape((offset - face_test, landmarks_num, 2))
    wct_landmarks = wct_landmarks.reshape((offset - face_test, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    if not os.path.exists(face_path.wct_landmarks):
        os.mkdir(face_path.wct_landmarks)
        print('Create dir: [{}]'.format(face_path.wct_landmarks))
    if not os.path.exists(face_path.wct_img):
        os.mkdir(face_path.wct_img)
        print('Create dir: [{}]'.format(face_path.wct_img))
    if not os.path.exists(face_path.wct_cmp):
        os.mkdir(face_path.wct_cmp)
        print('Create dir: [{}]'.format(face_path.wct_cmp))
    if not os.path.exists(face_path.wct_psm):
        os.mkdir(face_path.wct_psm)
        print('Create dir: [{}]'.format(face_path.wct_psm))

    if not os.path.exists(face_path.wct_mask):
        os.mkdir(face_path.wct_mask)
        print('Create dir: [{}]'.format(face_path.wct_mask))

    if not os.path.exists(face_path.wct_color):
        os.mkdir(face_path.wct_color)
        print('Create dir: [{}]'.format(face_path.wct_color))

    pairs = []
    for idx, face_name in enumerate(test_face_landmarks_names):
        pairs.append(
            (test_face_landmarks, test_wct_landmarks, face_path, face_name, idx, enable_draw, face_scale, cari_scale))

    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)


def make_wct_landmarks(face_path: DatasetPath, cari_path: DatasetPath,
                       offset, face_test, device_id,
                       landmarks_num=272, enable_draw=False, face_scale=0.5, cari_scale=2):
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'
    face_landmarks = []
    test_face_landmarks = []
    cari_landmarks = []

    total_landmarks_names = get_filenames(face_path.landmarks, offset)
    face_landmarks_names = total_landmarks_names[0:face_test]
    test_face_landmarks_names = total_landmarks_names[face_test:]
    cari_landmarks_names = get_filenames(cari_path.landmarks)

    # print(face_landmarks_names)

    print('Loading face landmarks from [{}]'.format(face_path.landmarks))
    for face_landmark_name in face_landmarks_names:
        print('Loading face [{}]...'.format(face_landmark_name))
        face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)))

    for test_face_landmark_name in test_face_landmarks_names:
        # face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)).reshape(-1, 1))
        print('Loading face [{}]...'.format(test_face_landmark_name))
        test_face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, test_face_landmark_name)))

    print('Loading face landmarks done.')

    print('Loading cari landmarks from [{}]'.format(cari_path.landmarks))
    for cari_landmark_name in cari_landmarks_names:
        # cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)).reshape(-1, 1))
        print('Loading cari [{}]...'.format(cari_landmark_name))
        cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)))
    print('Loading cari landmarks done.')

    # face_landmarks = np.array(face_landmarks, dtype=np.int).reshape((offset, landmarks_num, 2))
    # cari_landmarks = np.array(cari_landmarks, dtype=np.int).reshape((len(cari_landmarks_names), landmarks_num, 2))
    face_landmarks = (np.array(face_landmarks) * face_scale).astype(np.int)
    test_face_landmarks = (np.array(test_face_landmarks) * face_scale).astype(np.int)
    cari_landmarks = (np.array(cari_landmarks) * cari_scale).astype(np.int)

    fl_tensor = torch.from_numpy(face_landmarks).view(offset - face_test, -1).permute(1, 0).double().to(device)
    test_fl_tensor = torch.from_numpy(test_face_landmarks).view(face_test, -1).permute(1, 0).double().to(device)
    cl_tensor = torch.from_numpy(cari_landmarks).view(len(cari_landmarks_names), -1).permute(1, 0).double().to(device)

    # print(fl_tensor.size())
    # print(cl_tensor.size())

    start = time.time()
    print('Processing whiting and color operation: Total content features:[{}]/ style feature: [{}]'.format(
        len(face_landmarks_names), len(cari_landmarks_names)))
    wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    face_landmarks = face_landmarks.reshape((offset - face_test, landmarks_num, 2))
    wct_landmarks = wct_landmarks.reshape((offset - face_test, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    if not os.path.exists(face_path.wct_landmarks):
        os.mkdir(face_path.wct_landmarks)
        print('Create dir: [{}]'.format(face_path.wct_landmarks))
    if not os.path.exists(face_path.wct_img):
        os.mkdir(face_path.wct_img)
        print('Create dir: [{}]'.format(face_path.wct_img))
    if not os.path.exists(face_path.wct_cmp):
        os.mkdir(face_path.wct_cmp)
        print('Create dir: [{}]'.format(face_path.wct_cmp))
    if not os.path.exists(face_path.wct_psm):
        os.mkdir(face_path.wct_psm)
        print('Create dir: [{}]'.format(face_path.wct_psm))

    if not os.path.exists(face_path.wct_mask):
        os.mkdir(face_path.wct_mask)
        print('Create dir: [{}]'.format(face_path.wct_mask))

    if not os.path.exists(face_path.wct_color):
        os.mkdir(face_path.wct_color)
        print('Create dir: [{}]'.format(face_path.wct_color))

    pairs = []
    for idx, face_name in enumerate(test_face_landmarks_names):
        pairs.append(
            (test_face_landmarks, test_wct_landmarks, face_path, face_name, idx, enable_draw, face_scale, cari_scale))

    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)


def split_wct_landmarks(web_cari_path, landmark_path, offset, face_test, device_id
                        , landmarks_num=272, enable_draw=False, face_scale=0.5, cari_scale=2):
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'
    face_landmarks = []
    test_face_landmarks = []
    cari_landmarks = []

    chars = os.listdir(web_cari_path)

    random_half_face_chars = random.sample(chars, len(chars) / 2)

    total_landmarks_names = get_filenames(face_path.landmarks, offset)
    face_landmarks_names = total_landmarks_names[0:face_test]
    test_face_landmarks_names = total_landmarks_names[face_test:]
    cari_landmarks_names = get_filenames(cari_path.landmarks)

    # print(face_landmarks_names)

    print('Loading face landmarks from [{}]'.format(face_path.landmarks))
    for face_landmark_name in face_landmarks_names:
        print('Loading face [{}]...'.format(face_landmark_name))
        face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)))

    for test_face_landmark_name in test_face_landmarks_names:
        # face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, face_landmark_name)).reshape(-1, 1))
        print('Loading face [{}]...'.format(test_face_landmark_name))
        test_face_landmarks.append(np.loadtxt(os.path.join(face_path.landmarks, test_face_landmark_name)))

    print('Loading face landmarks done.')

    print('Loading cari landmarks from [{}]'.format(cari_path.landmarks))
    for cari_landmark_name in cari_landmarks_names:
        # cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)).reshape(-1, 1))
        print('Loading cari [{}]...'.format(cari_landmark_name))
        cari_landmarks.append(np.loadtxt(os.path.join(cari_path.landmarks, cari_landmark_name)))
    print('Loading cari landmarks done.')

    # face_landmarks = np.array(face_landmarks, dtype=np.int).reshape((offset, landmarks_num, 2))
    # cari_landmarks = np.array(cari_landmarks, dtype=np.int).reshape((len(cari_landmarks_names), landmarks_num, 2))
    face_landmarks = (np.array(face_landmarks) * face_scale).astype(np.int)
    test_face_landmarks = (np.array(test_face_landmarks) * face_scale).astype(np.int)
    cari_landmarks = (np.array(cari_landmarks) * cari_scale).astype(np.int)

    fl_tensor = torch.from_numpy(face_landmarks).view(offset - face_test, -1).permute(1, 0).double().to(device)
    test_fl_tensor = torch.from_numpy(test_face_landmarks).view(face_test, -1).permute(1, 0).double().to(device)
    cl_tensor = torch.from_numpy(cari_landmarks).view(len(cari_landmarks_names), -1).permute(1, 0).double().to(device)

    # print(fl_tensor.size())
    # print(cl_tensor.size())

    start = time.time()
    print('Processing whiting and color operation: Total content features:[{}]/ style feature: [{}]'.format(
        len(face_landmarks_names), len(cari_landmarks_names)))
    wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    face_landmarks = face_landmarks.reshape((offset - face_test, landmarks_num, 2))
    wct_landmarks = wct_landmarks.reshape((offset - face_test, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    if not os.path.exists(face_path.wct_landmarks):
        os.mkdir(face_path.wct_landmarks)
        print('Create dir: [{}]'.format(face_path.wct_landmarks))
    if not os.path.exists(face_path.wct_img):
        os.mkdir(face_path.wct_img)
        print('Create dir: [{}]'.format(face_path.wct_img))
    if not os.path.exists(face_path.wct_cmp):
        os.mkdir(face_path.wct_cmp)
        print('Create dir: [{}]'.format(face_path.wct_cmp))
    if not os.path.exists(face_path.wct_psm):
        os.mkdir(face_path.wct_psm)
        print('Create dir: [{}]'.format(face_path.wct_psm))

    if not os.path.exists(face_path.wct_mask):
        os.mkdir(face_path.wct_mask)
        print('Create dir: [{}]'.format(face_path.wct_mask))

    if not os.path.exists(face_path.wct_color):
        os.mkdir(face_path.wct_color)
        print('Create dir: [{}]'.format(face_path.wct_color))

    pairs = []
    # pm_pairs = []
    for idx, face_name in enumerate(test_face_landmarks_names):
        pairs.append(
            (test_face_landmarks, test_wct_landmarks, face_path, face_name, idx, enable_draw, face_scale, cari_scale))
        # pairs.append(
        #     (face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw, face_scale, cari_scale))

    # index = os.path.splitext(face_name)[0]
    # pm_pairs.append((index, index, face_landmarks[idx], wct_landmarks[idx], face_path.wct_psm))

    # p = mp.Pool(mp.cpu_count() / 2)
    # with mp.Pool(mp.cpu_count() - 1) as p:
    #     p.starmap(wct_warp, pairs)

    # with mp.Pool(mp.cpu_count() - 1) as p:
    #     p.starmap(warp_position_map_by_kpts_task, pm_pairs)

    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)
    # for idx, face_name in enumerate(face_landmarks_names):
    #         p.apply_async(wct_task, (face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw,))
    # with mp.Pool(mp.cpu_count() - 1) as p:
    # for idx, face_name in enumerate(face_landmarks_names):
    #     p.apply_async(wct_task, (face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw,))
    # for idx, face_name in enumerate(face_landmarks_names):
    #     wct_task(face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw)

    # warp_position_map_by_kpts_task(os.path.splitext(face_name)[0], index, face_landmarks[idx], wct_landmarks[idx],
    #                                face_path.wct_psm)
    # face_names = get_filenames(FACE_IMG_PATH, offset)


def wct_task(face_landmarks, wct_landmarks, face_path: DatasetPath, face_name, idx, enable_draw, face_scale=0.5,
             cari_scale=2):
    index = os.path.splitext(face_name)[0]
    warped_path = os.path.join(face_path.wct_img, index + '.jpg')
    landmark_save_path = os.path.join(face_path.wct_landmarks, index + '.txt')
    cmp_save_path = os.path.join(face_path.wct_cmp, index + '.jpg')
    face = cv2.resize(cv2.imread(os.path.join(face_path.img, index + '.jpg')), (0, 0),
                      fx=face_scale, fy=face_scale)
    if face is None:
        raise Exception('Empty img [{}].'.format(face_name))
    print('Processing warping based WCT landmarks: [{}]'.format(face_name))
    warped, transform = warp_image(face, face_landmarks[idx], wct_landmarks[idx])
    warped = (warped * 255).astype(np.uint8)

    cv2.imwrite(warped_path, warped)
    np.savetxt(landmark_save_path, wct_landmarks[idx], fmt='%d')
    print('Processed warping done based WCT landmarks: [{}]'.format(face_name))

    mask_name = index + '.png'
    face_mask = cv2.imread(os.path.join(face_path.mask, mask_name), cv2.IMREAD_GRAYSCALE)
    cw_cmp = None
    if face_mask is not None and enable_draw:
        print('Warping mask: [{}]'.format(mask_name))
        face_color = colormap[face_mask].astype(np.uint8)
        warped_mask, warped_mask_color = warped_color_nearest(face_landmarks[idx], wct_landmarks[idx], face_color)
        print('Done Warping mask: [{}]'.format(mask_name))
        cw_cmp = cv2.cvtColor(np.hstack((face_color, warped_mask_color)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(face_path.wct_mask, mask_name), warped_mask)
        cv2.imwrite(os.path.join(face_path.wct_color, mask_name), cv2.cvtColor(warped_mask_color, cv2.COLOR_RGB2BGR))
    # warped_mask, warped_mask_color = warp_nearest(face_landmarks[idx], wct_landmarks[idx], face_color)
    # index, transform, wapred = wct_warp(face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw)
    if enable_draw:
        warped_kpts = draw_key_points(warped, wct_landmarks[idx])
        face = draw_key_points(face, face_landmarks[idx])
        fw_cmp = np.hstack((face, warped_kpts))
        if cw_cmp is not None:
            cmp = np.vstack((fw_cmp, cw_cmp))
        else:
            cmp = fw_cmp
        cv2.imwrite(cmp_save_path, cmp)
    print('Done warping mask: [{}]'.format(mask_name))

    # warp_position_map_by_kpts_task(index, index, face_landmarks[idx], wct_landmarks[idx], face_path.wct_psm)


# def wct_warp(face_landmarks, wct_landmarks, face_path, face_name, idx, enable_draw):
#     print('Face landmarks size: [{}]'.format(face_landmarks.shape))
#     print('WCT landmarks size: [{}]'.format(wct_landmarks.shape))
#
# return index, transform, warped


def warp_position_map_by_kpts_task(f_index, c_index, fpkts, cpkts, save_path):
    # print('Processing face [{}] and cari [{}] position map calculation'.format(f_index, c_index))
    # f_index = os.path.splitext(ff)[0]
    # c_index = os.path.splitext(cf)[0]
    # savetxt_name = f_index + '-' + c_index + '.txt'
    print('Esitimating [{}]-[{}] position map....'.format(f_index, c_index))
    offset_filed = estimate_offset_field_by_kpts(fpkts, cpkts, IMG_SIZE, IMG_SIZE, x_position_map,
                                                 y_position_map)
    offset_filed = offset_filed.reshape(IMG_SIZE, -1).numpy()
    save_npy(c_index, f_index, offset_filed, save_path)
    print('Done face [{}] and cari [{}] position map calculation'.format(f_index, c_index))
    return True


def generate_img_with_landmarks(path, landmarks_path, save_path, scale=0.5, offset=None):
    # file_names = sorted_names(landmarks_path)
    file_names = get_filenames(landmarks_path, offset)
    if len(file_names) <= 0:
        msg = 'No file exists in [{}].'.format(path)
        raise Exception(msg)

    if not os.path.exists(save_path):
        print('Make new dir: [{}]'.format(save_path))
        os.mkdir(save_path)

    # process_pool = mp.Pool(processes=mp.cpu_count() - 1)
    # for name in file_names:
    #     process_pool.apply_async(landmarks_img_task, args=(path, landmarks_path, save_path, name, scale))
    # process_pool.close()
    # process_pool.join()
    args = [(path, landmarks_path, save_path, name, scale) for name in file_names]
    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(landmarks_img_task, args)


def landmarks_img_task(path, landmarks_path, save_path, name, scale):
    filename, extention = os.path.splitext(name)
    if not is_img_format(extention):
        print('Filtered file [{}].'.format(name))
        return False
    print('Processing [{}]...........'.format(name))
    img_path = os.path.join(path, name)
    img = cv2.imread(img_path)
    if img is None:
        print('Loading img [{}] Error, emtpy img.'.format(name))
        return True
    key_points = np.loadtxt(os.path.join(landmarks_path, filename + '.txt'), dtype=np.int) * scale
    key_points = key_points.astype(np.int)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img = draw_key_points(img, key_points)
    cv2.imwrite(os.path.join(save_path, filename + '.png'), img)
    print('Processed [{}].........'.format(name))


def off_line_cal_warp(face_landmarks_path, cari_landmarks_path, save_path, txt_path=None, face_offset=None,
                      cari_offset=None):
    face_filenames = get_filenames(face_landmarks_path, face_offset)
    cari_filenames = get_filenames(cari_landmarks_path, cari_offset)

    if not os.path.exists(save_path):
        print('Make new dir: [{}]'.format(save_path))
        os.mkdir(save_path)
    pairs = []
    for ff in face_filenames:
        for cf in cari_filenames:
            pairs.append(
                (os.path.join(face_landmarks_path, ff), os.path.join(cari_landmarks_path, cf), ff, cf, save_path,
                 txt_path))

    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(warp_position_map_task, pairs)


def warp_position_map_task(afp, acp, ff, cf, save_path, txt_path=None):
    print('Processing face [{}] and cari [{}]'.format(ff, cf))
    f_index = os.path.splitext(ff)[0]
    c_index = os.path.splitext(cf)[0]

    if txt_path is not None:
        savetxt_name = f_index + '-' + c_index + '.txt'
        txt = os.path.join(txt_path, savetxt_name)
        if os.path.exists(txt):
            offset_filed = np.loadtxt(txt)
            print('Load from exisit txt: [{}]'.format(savetxt_name))
            save_npy(c_index, f_index, offset_filed, save_path)
            return True
        else:
            fpkts = np.loadtxt(afp, dtype=np.int)
            cpkts = np.loadtxt(acp, dtype=np.int)
            print('Esitimating [{}]-[{}]....'.format(f_index, c_index))
            offset_filed = estimate_offset_field_by_kpts(fpkts, cpkts, IMG_SIZE, IMG_SIZE, x_position_map,
                                                         y_position_map)
            offset_filed = offset_filed.reshape(IMG_SIZE, -1).numpy()
            # save_txt(c_index, f_index, offset_filed, save_path)
            # save_npy(c_index, f_index, offset_filed, save_path)
            save_npy(c_index, f_index, offset_filed, save_path)
            # save_pickle(c_index, f_index, offset_filed, save_path)
            print('Processed face [{}] and cari [{}]'.format(ff, cf))
            return True


def save_bin(c_index, f_index, offset_filed, save_path):
    # bin_time = time.time()
    save_name = f_index + '-' + c_index + '.bin'
    offset_filed.tofile(os.path.join(save_path, save_name))
    # print('Bin need time [{}]'.format(time.time() - bin_time))


def save_npy(c_index, f_index, offset_filed, save_path):
    # npy_st = time.time()
    save_name = f_index + '-' + c_index + '.npy'
    np.save(os.path.join(save_path, save_name), offset_filed)
    # print('Npy need time [{}]'.format(time.time() - npy_st))


def save_txt(c_index, f_index, offset_filed, save_path):
    # txt_st = time.time()
    save_name = f_index + '-' + c_index + '.txt'
    np.savetxt(os.path.join(save_path, save_name), offset_filed, fmt='%.5f')
    # print('Txt need time [{}]'.format(time.time() - txt_st))


def save_pickle(c_index, f_index, offset_filed, save_path):
    pkl_time = time.time()
    save_name = f_index + '-' + c_index + '.pkl'
    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump(offset_filed, f)
    print('PKL∂ need time [{}]'.format(time.time() - pkl_time))


def warp_face(path, face_path, cari_path, save_path):
    # face_filenames = get_filenames(face_landmarks_path, face_offset)
    # cari_filenames = get_filenames(cari_landmarks_path, cari_offset)
    filenames = get_filenames(path)
    if not os.path.exists(save_path):
        print('Make new dir: [{}]'.format(save_path))
        os.mkdir(save_path)
    pairs = []
    for p in filenames:
        splited = p.split('-')
        face_index, cari_index = splited[0], splited[1]
        pair = (os.path.join(face_path, face_index + '.jpg', os.path.join(cari_path, cari_index + '.jpg')))
        pairs.append(pair)

    # for ff in face_filenames:
    #     for cf in cari_filenames:
    #         pairs.append(
    #             (os.path.join(face_landmarks_path, ff), os.path.join(cari_landmarks_path, cf), ff, cf, save_path,
    #              txt_path))

    # with mp.Pool(mp.cpu_count() - 1) as p:
    #     p.starmap(warp_face_task, pairs)


# def txt_to_npy(warp_psm_path):
#     print('Processing '.format(ff, cf))
#     print('Processing face [{}] and cari [{}]'.format(ff, cf))

def colorize_mask(path):
    filenames = get_filenames(path.mask)

    if not os.path.exists(path.color):
        print('Create dir: [{}]'.format(path.wct_img))
        os.mkdir(path.color)

    for mask_name in filenames:
        mask = cv2.imread(os.path.join(path.mask, mask_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise Exception('Mask [{}] is empty.'.format(mask_name))
        print('Processing: [{}]'.format(mask_name))
        color_mask = colormap[mask].astype(np.uint8)
        mask_name = os.path.splitext(mask_name)[0] + '.png'
        cv2.imwrite(os.path.join(path.color, mask_name), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        print('Done: [{}]'.format(mask_name))


def draw_key_points_in_img(path: DatasetPath, scale=1, offset=None):
    if offset is not None:
        names = get_filenames(path.img, offset)
    else:
        names = get_filenames(path.img)
    if not os.path.exists(path.landmarks_img):
        os.mkdir(path.landmarks_img)
    for n in names:
        prefix = os.path.splitext(n)[0]
        landmark_name = prefix + '.txt'
        landmark_path = os.path.join(path.landmarks, landmark_name)
        print(landmark_path)
        if os.path.exists(landmark_path):
            landmarks = np.loadtxt(landmark_path, dtype=np.int)
            print(os.path.join(path.img, n))
            img = cv2.imread(os.path.join(path.img, n))
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            if img is not None:
                img = draw_key_points(img, landmarks)
                cv2.imwrite(os.path.join(path.landmarks_img, n), img)


def main():
    # make_landmarks(CARI_DATASET_PATH, CARI_LANDMARKS_PATH)
    # generate_img_with_landmarks(CARI_IMG_PATH, CARI_LANDMARKS_PATH, CARI_LANDMARKS_IMG_PATH, scale=1)
    # make_landmarks(FACE_DATASET_PATH, FACE_LANDMARKS_PATH)
    # generate_img_with_landmarks(FACE_IMG_PATH, FACE_LANDMARKS_PATH, FACE_LANDMARKS_IMG_PATH, scale=0.5)
    # off_line_cal_warp(FACE_LANDMARKS_PATH, CARI_LANDMARKS_PATH, SERVER_WARPED_PSM_PATH, WARP_PSM_PATH,
    #                   face_offset=FACE_SAMPLE_OFFSET)
    # off_line_cal_warp(FACE_LANDMARKS_PATH, CARI_LANDMARKS_PATH, WARP_PSM_PATH)
    # generate_name_list(FACE_WCT_IMG_PATH, offset=20000)
    # generate_name_list(CARI_PATH)
    # make_wct_landmarks(celeb_path, cari_path, 20000, 2, enable_draw=True, landmarks_num=845)
    make_wct_landmarks(celeb_path, a1_path, 1000, 500, 0, enable_draw=True, landmarks_num=845, face_scale=0.5,
                       cari_scale=2)
    # make_wct_landmarks(celeb_path, cari_path, 500, 0, enable_draw=True, landmarks_num=845, scale=0.5)
    # draw_key_points_in_img(a1_path, 500
    # )
    # colorize_mask(cari_path)


if __name__ == '__main__':
    main()
