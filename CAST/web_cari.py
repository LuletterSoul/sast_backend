#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: web_cari.py
@time: 3/21/20 5:06 PM
@version 1.0
@desc:
"""
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from random import sample

import cv2
import numpy as np
import torch
from utils.wct import whiten_and_color, warp_image, draw_key_points, whiten_and_color_mu


def cat(input_dir: Path, web_cari_img: Path, random_train_nums, sample_num=3, rows=3, random_sample_times=2):
    test_dir = input_dir / 'record' / 'test.txt'
    warp_dir = input_dir / 'warp'
    plot_output_dir = input_dir / 'plot'
    if not plot_output_dir.exists():
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(test_dir)) as f:
        names = [os.path.basename(path.replace('\n', '')) for path in f.readlines()]

    plot = None
    for idx, n in enumerate(names):
        # fetch original photos
        samples = sample(list(Path(web_cari_img / n).glob('P*.jpg')), sample_num)  # only sample some photos
        for s in samples:
            row_imgs = [cv2.imread(str(s))]
            for rwt in random_train_nums:
                rw = str(warp_dir / str(rwt) / n.replace(' ', '_') / (
                        os.path.splitext(os.path.basename(str(s)))[0] + '.png'))
                cwd = cv2.imread(rw)
                if cwd is None:
                    continue
                row_imgs.append(cwd)
            if plot is None:
                plot = np.hstack(row_imgs)
                print(plot.shape)
            else:
                if len(row_imgs) * 256 == plot.shape[1]:
                    plot = np.vstack([plot, np.hstack(row_imgs)])
                    print(plot.shape)
        if (idx + 1) % rows == 0:
            cv2.imwrite(str(plot_output_dir / f'{str(idx)}.png'), plot)
            plot = None
    if plot is not None:
        cv2.imwrite(str(plot_output_dir / f'{str(len(names))}.png'), plot)


def cat_from_two_times(input_dir: Path, web_cari_img: Path, random_train_nums, sample_num=3, rows=3, fmt='.bmp'):
    test_dir = input_dir / 'record' / 'test.txt'
    warp_dir = input_dir / 'warp'
    plot_output_dir = input_dir / 'plot'
    if not plot_output_dir.exists():
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(test_dir)) as f:
        names = [os.path.basename(path.replace('\n', '')) for path in f.readlines()]

    plot = None
    for idx, n in enumerate(names):
        # fetch original photos
        face_lists = list(Path(web_cari_img / n).glob(f'P*{fmt}'))
        if len(face_lists) < sample_num:
            continue
        samples = sample(face_lists, sample_num)  # only sample some photos
        for s in samples:
            row_imgs = [cv2.imread(str(s))]
            for rwt in random_train_nums:
                rw = str(warp_dir / f'0-{str(rwt)}' / n.replace(' ', '_') / (
                        os.path.splitext(os.path.basename(str(s)))[0] + '.png'))
                cwd = cv2.imread(rw)
                if cwd is None:
                    continue
                row_imgs.append(cwd)
                if rwt == 20:
                    rw = str(warp_dir / f'1-{str(rwt)}' / n.replace(' ', '_') / (
                            os.path.splitext(os.path.basename(str(s)))[0] + '.png'))
                    cwd = cv2.imread(rw)
                    row_imgs.append(cwd)
            if plot is None:
                plot = np.hstack(row_imgs)
                print(plot.shape)
            else:
                if len(row_imgs) * 256 == plot.shape[1]:
                    plot = np.vstack([plot, np.hstack(row_imgs)])
                    print(plot.shape)
        if (idx + 1) % rows == 0:
            cv2.imwrite(str(plot_output_dir / f'{str(idx)}.png'), plot)
            plot = None
    if plot is not None:
        cv2.imwrite(str(plot_output_dir / f'{str(len(names))}.png'), plot)


def warp_ablation2(web_cari_path: Path, random_train_nums, test_times, device_id
                   , landmarks_num, warped_output, cmp_output, record_output, scale=-1, scale_to_old=False,
                   rollback=False, random_sample_times=1, fmt='.jpg', draw_pts=False, train_face_scale=1,
                   train_cari_scale=1, test_scale=1):
    """
    :param web_cari_path:
    :param random_train_nums:
    :param test_times:
    :param device_id:
    :param landmarks_num:
    :param warped_output:
    :param cmp_output:
    :return:
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    if not record_output.exists():
        record_output.mkdir(parents=True, exist_ok=True)

    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(landmarks_num)

    # cari_train_photo_path = cari_train_path / 'img'
    # cari_train_landmarks_path = cari_train_path / 'landmarks'

    # photo_paths = list(web_cari_photo_path.glob('*'))
    # cari_paths = list(web_cari_photo_path.glob('*'))

    photo_people_paths = list(web_cari_photo_path.glob('*'))
    cari_people_paths = list(web_cari_photo_path.glob('*'))
    face_test_paths = []

    print(f'Photo num {len(photo_people_paths)}')
    print(f'Cari num {len(cari_people_paths)}')

    if not rollback:
        train_people_paths = photo_people_paths[:len(photo_people_paths) // 2]
        test_people_paths = photo_people_paths[len(photo_people_paths) // 2:]
        np.random.shuffle(test_people_paths)
        # select samples of 126 peoples as test set.
        with open(f'{record_output}/test.txt', 'w') as fw:
            for path in test_people_paths:
                face_test_paths.extend(list(path.glob(f'P*{fmt}')))
                fw.write(str(path) + '\n')
    else:
        with open(f'{record_output}/test.txt') as f:
            rollback_test_list = [path.replace('\n', '') for path in f.readlines()]
            train_people_paths = [p for p in photo_people_paths if str(p) not in rollback_test_list]
            print(train_people_paths)
            for path in rollback_test_list:
                face_test_paths.extend(list(Path(path).glob(f'P*{fmt}')))

    for n in range(random_sample_times):
        np.random.shuffle(train_people_paths)
        photo_paths = []
        cari_paths = []
        # select 1,,5,10,20 people from training set randomly.
        for rtn in random_train_nums:
            photo_path = []
            cari_path = []
            with open(f'{record_output}/{rtn}.txt', 'w') as fw:
                for path in train_people_paths:
                    plist = list(path.glob(f'P*{fmt}'))
                    clist = list(path.glob(f'C*{fmt}'))
                    if len(plist) >= rtn:
                        photo_path.extend(sample(plist, rtn))
                    else:
                        photo_path.extend(plist)
                    if len(clist) >= rtn:
                        cari_path.extend(sample(clist, rtn))
                    else:
                        cari_path.extend(clist)
                for p in photo_path:
                    fw.write(str(p) + '\n')
            photo_paths.append(photo_path)
            cari_paths.append(cari_path)

        # loads face test landmarks
        print('Loading face test landmarks.')
        test_face_landmarks, face_test_paths = load_landmarks(face_test_paths, web_cari_landmarks_path)

        test_face_landmarks = test_face_landmarks * test_scale

        print('Loading face test landmarks done.')
        face_test_num = len(test_face_landmarks)
        test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks)

        for idx, p in enumerate(photo_paths):
            print(f'Processing sample people num : {random_train_nums[idx]}')
            photo_path = photo_paths[idx]
            cari_path = cari_paths[idx]
            # with open(f'{record_output}/{str(random_train_nums[idx])}.txt', 'w') as fw:
            #     for path in p:
            #         photo_paths.extend(list(path.glob(f'P*{fmt}')))
            #         cari_paths.extend(list(path.glob(f'C*{fmt}')))
            #         fw.write(str(path) + '\n')
            print(f'Photo num: {len(photo_path)}')
            print(f'Cari num: {len(cari_path)}')
            train_face_landmarks, face_train_paths = load_landmarks(photo_path, web_cari_landmarks_path)
            cari_landmarks, cari_train_paths = load_landmarks(cari_path, web_cari_landmarks_path)

            train_face_landmarks = train_face_landmarks * train_face_scale
            cari_landmarks = cari_landmarks * train_cari_scale

            # training sample num
            face_train_num = len(train_face_landmarks)
            cari_train_num = len(cari_landmarks)

            fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks)
            cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks)
            # loads cari training landmarks

            start = time.time()
            print(
                f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
            # perform WCT feature transformation
            wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
            # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
            print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

            # train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
            # train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
            # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
            test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
            test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num,
                                                                       2).long().cpu().numpy()
            print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
            print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
            # pm_pairs = []

            current_epoch_warped_output = warped_output / f'{str(n)}-{str(random_train_nums[idx])}'
            current_epoch_cmp_output = cmp_output / f'{str(n)}-{str(random_train_nums[idx])}'
            if not current_epoch_warped_output.exists():
                warped_output.mkdir(parents=True, exist_ok=True)
            if not current_epoch_cmp_output.exists():
                cmp_output.mkdir(parents=True, exist_ok=True)

            pairs = []
            for idx, fp in enumerate(face_test_paths):
                pairs.append(
                    (
                        test_face_landmarks[idx], test_wct_landmarks[idx], fp, current_epoch_warped_output,
                        current_epoch_cmp_output, scale,
                        scale_to_old, False, test_scale,))
            # acceleration warping using multiprocessing
            with mp.Pool(mp.cpu_count() - 1) as p:
                p.starmap(wct_task, pairs)


def warp_ablation(web_cari_path: Path, random_train_nums, test_times, device_id
                  , landmarks_num, warped_output, cmp_output, record_output, scale=-1, scale_to_old=False,
                  rollback=False, random_sample_times=1, fmt='.jpg'):
    """
    use 5,10,15,20 ... peoples to execute warping generation
    :param web_cari_path:
    :param random_train_nums:
    :param test_times:
    :param device_id:
    :param landmarks_num:
    :param warped_output:
    :param cmp_output:
    :return:
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    if not record_output.exists():
        record_output.mkdir(parents=True, exist_ok=True)

    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(landmarks_num)

    # cari_train_photo_path = cari_train_path / 'img'
    # cari_train_landmarks_path = cari_train_path / 'landmarks'

    # photo_paths = list(web_cari_photo_path.glob('*'))
    # cari_paths = list(web_cari_photo_path.glob('*'))

    photo_people_paths = list(web_cari_photo_path.glob('*'))
    cari_people_paths = list(web_cari_photo_path.glob('*'))
    face_test_paths = []

    print(f'Photo num {len(photo_people_paths)}')
    print(f'Cari num {len(cari_people_paths)}')

    if not rollback:
        train_people_paths = photo_people_paths[:len(photo_people_paths) // 2]
        test_people_paths = photo_people_paths[len(photo_people_paths) // 2:]
        np.random.shuffle(test_people_paths)
        # select samples of 126 peoples as test set.
        with open(f'{record_output}/test.txt', 'w') as fw:
            for path in test_people_paths:
                face_test_paths.extend(list(path.glob(f'P*{fmt}')))
                fw.write(str(path) + '\n')
    else:
        with open(f'{record_output}/test.txt') as f:
            rollback_test_list = [path.replace('\n', '') for path in f.readlines()]
            train_people_paths = [p for p in photo_people_paths if str(p) not in rollback_test_list]
            print(train_people_paths)
            for path in rollback_test_list:
                face_test_paths.extend(list(Path(path).glob(f'P*{fmt}')))

    for n in range(random_sample_times):
        np.random.shuffle(train_people_paths)
        # select 1,,5,10,20 people from training set randomly.
        sample_people_paths = [sample(train_people_paths, rtn) for rtn in random_train_nums]

        # loads face test landmarks
        print('Loading face test landmarks.')
        test_face_landmarks, face_test_paths = load_landmarks(face_test_paths, web_cari_landmarks_path)

        print('Loading face test landmarks done.')
        face_test_num = len(test_face_landmarks)
        test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks)

        for idx, p in enumerate(sample_people_paths):
            print(f'Processing sample people num : {random_train_nums[idx]}')
            photo_paths = []
            cari_paths = []
            with open(f'{record_output}/{str(random_train_nums[idx])}.txt', 'w') as fw:
                for path in p:
                    photo_paths.extend(list(path.glob(f'P*{fmt}')))
                    cari_paths.extend(list(path.glob(f'C*{fmt}')))
                    fw.write(str(path) + '\n')
            print(len(photo_paths))
            train_face_landmarks, face_train_paths = load_landmarks(photo_paths, web_cari_landmarks_path)
            cari_landmarks, cari_train_paths = load_landmarks(cari_paths, web_cari_landmarks_path)

            # training sample num
            face_train_num = len(train_face_landmarks)
            cari_train_num = len(cari_landmarks)

            fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks)
            cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks)
            # loads cari training landmarks

            start = time.time()
            print(
                f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
            # perform WCT feature transformation
            wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
            # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
            print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

            # train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
            # train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
            # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
            test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
            test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num,
                                                                       2).long().cpu().numpy()
            print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
            print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
            # pm_pairs = []

            current_epoch_warped_output = warped_output / f'{str(n)}-{str(random_train_nums[idx])}'
            current_epoch_cmp_output = cmp_output / f'{str(n)}-{str(random_train_nums[idx])}'
            if not current_epoch_warped_output.exists():
                warped_output.mkdir(parents=True, exist_ok=True)
            if not current_epoch_cmp_output.exists():
                cmp_output.mkdir(parents=True, exist_ok=True)

            pairs = []
            for idx, fp in enumerate(face_test_paths):
                pairs.append(
                    (
                        test_face_landmarks[idx], test_wct_landmarks[idx], fp, current_epoch_warped_output,
                        current_epoch_cmp_output, scale,
                        scale_to_old,))
            # acceleration warping using multiprocessing
            with mp.Pool(mp.cpu_count() - 1) as p:
                p.starmap(wct_task, pairs)


def warp_fixed(web_cari_path: Path, fixed_faces_num, fixed_caris_num, device_id
               , landmarks_num, warped_output, cmp_output, record_output, scale=-1, scale_to_old=False,
               rollback=False, random_sample_times=1, fmt='.jpg'):
    """
    :param web_cari_path:
    :param random_train_nums:
    :param device_id:
    :param landmarks_num:
    :param warped_output:
    :param cmp_output:
    :return:
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    if not record_output.exists():
        record_output.mkdir(parents=True, exist_ok=True)

    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(landmarks_num)

    photo_people_paths = list(web_cari_photo_path.glob('*'))
    cari_people_paths = list(web_cari_photo_path.glob('*'))
    face_test_paths = []

    print(f'Photo num {len(photo_people_paths)}')
    print(f'Cari num {len(cari_people_paths)}')

    if not rollback:
        train_people_paths = photo_people_paths[:len(photo_people_paths) // 2]
        test_people_paths = photo_people_paths[len(photo_people_paths) // 2:]
        np.random.shuffle(test_people_paths)
        # select samples of 126 peoples as test set.
        with open(f'{record_output}/test.txt', 'w') as fw:
            for path in test_people_paths:
                face_test_paths.extend(list(path.glob(f'P*{fmt}')))
                fw.write(str(path) + '\n')
    else:
        with open(f'{record_output}/test.txt') as f:
            rollback_test_list = [path.replace('\n', '') for path in f.readlines()]
            train_people_paths = [p for p in photo_people_paths if str(p) not in rollback_test_list]
            print(train_people_paths)
            for path in rollback_test_list:
                face_test_paths.extend(list(Path(path).glob(f'P*{fmt}')))

    for n in range(random_sample_times):
        np.random.shuffle(train_people_paths)
        # loads face test landmarks
        print('Loading face test landmarks.')
        test_face_landmarks, face_test_paths = load_landmarks(face_test_paths, web_cari_landmarks_path)

        print('Loading face test landmarks done.')
        face_test_num = len(test_face_landmarks)
        test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks)

        photo_paths = []
        cari_paths = []
        for path in train_people_paths:
            photo_paths.extend(list(path.glob(f'P*{fmt}')))
            cari_paths.extend(list(path.glob(f'C*{fmt}')))
        photo_paths = sample(photo_paths, fixed_faces_num)
        cari_paths = sample(cari_paths, fixed_caris_num)
        with open(f'{record_output}/train_face_samples.txt', 'w') as fw:
            for path in photo_paths:
                fw.write(str(path) + '\n')

        with open(f'{record_output}/train_cari_samples.txt', 'w') as fw:
            for path in cari_paths:
                fw.write(str(path) + '\n')

        print(f'Train photo num: {len(photo_paths)}')
        print(f'Train cari num: {len(cari_paths)}')

        train_face_landmarks, face_train_paths = load_landmarks(photo_paths, web_cari_landmarks_path)
        cari_landmarks, cari_train_paths = load_landmarks(cari_paths, web_cari_landmarks_path)

        # training sample num
        face_train_num = len(train_face_landmarks)
        cari_train_num = len(cari_landmarks)

        fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks)
        cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks)
        # loads cari training landmarks

        start = time.time()
        print(
            f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
        # perform WCT feature transformation
        wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
        # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
        print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

        # train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
        # train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
        # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
        test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
        test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num,
                                                                   2).long().cpu().numpy()
        print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
        print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
        # pm_pairs = []

        suffix = f'{str(fixed_faces_num)}-{str(fixed_caris_num)}'
        current_epoch_warped_output = warped_output / suffix
        current_epoch_cmp_output = cmp_output / suffix
        if not current_epoch_warped_output.exists():
            warped_output.mkdir(parents=True, exist_ok=True)
        if not current_epoch_cmp_output.exists():
            cmp_output.mkdir(parents=True, exist_ok=True)

        pairs = []
        for idx, fp in enumerate(face_test_paths):
            pairs.append(
                (
                    test_face_landmarks[idx], test_wct_landmarks[idx], fp, current_epoch_warped_output,
                    current_epoch_cmp_output, scale,
                    scale_to_old,))
        # acceleration warping using multiprocessing
        with mp.Pool(mp.cpu_count() - 1) as p:
            p.starmap(wct_task, pairs)


def warp_use_train_and_test(photo_path: Path, test_path, cari_train_path,
                            train_face_num, train_cari_num, device_id
                            , landmarks_num, warped_output, cmp_output, record_output, compress_face_distribution=False,
                            compress_cari_distribution=False, scale=-1, scale_to_old=False, fmt='.jpg', draw_pts=False,
                            face_scale=0.5, cari_scale=1, use_web_cari=True, load_from_file=False):
    """
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    if use_web_cari:
        photo_image_path = photo_path / 'img'
        photo_landmarks_path = photo_path / 'landmarks' / str(landmarks_num)
        photo_paths = list(photo_image_path.glob(f'*/P*{fmt}'))
    else:
        photo_image_path = photo_path / 'img'
        photo_landmarks_path = photo_path / 'landmarks'
        photo_paths = list(photo_image_path.glob('*'))

    test_photo_path = test_path / 'img'
    test_landmark_path = test_path / 'landmarks'

    cari_train_photo_path = cari_train_path / 'img'
    cari_train_photo_landmarks_path = cari_train_path / 'landmarks'

    face_test_paths = list(test_photo_path.glob('*'))
    cari_paths = list(cari_train_photo_path.glob('*'))

    np.random.shuffle(photo_paths)
    # np.random.shuffle(cari_paths)

    print(f'Photo num {len(photo_paths)}')
    print(f'Cari num {len(cari_paths)}')

    # loads face training landmarks
    print('Loading face training landmarks.')
    if load_from_file:
        with open(f'{record_output}/train_face_samples.txt') as f:
            face_train_paths = [Path(path.replace('\n', '')) for path in f.readlines()]
        with open(f'{record_output}/train_cari_samples.txt') as f:
            cari_paths = [Path(path.replace('\n', '')) for path in f.readlines()]
    else:
        if compress_face_distribution:
            face_train_paths = photo_paths[:train_face_num]
        else:
            face_train_paths = photo_paths
        if compress_cari_distribution:
            cari_paths = sample(cari_paths, train_cari_num)
        with open(f'{record_output}/train_face_samples.txt', 'w') as fw:
            for path in face_train_paths:
                fw.write(str(path) + '\n')
        with open(f'{record_output}/train_cari_samples.txt', 'w') as fw:
            for path in cari_paths:
                fw.write(str(path) + '\n')

    if use_web_cari:
        train_face_landmarks, face_paths = load_landmarks(face_train_paths, photo_landmarks_path)
    else:
        train_face_landmarks, face_paths = load_landmarks_from_single_path(face_train_paths, photo_landmarks_path)

    print('Loading face training landmarks done.')

    print('Loading cari training landmarks.')
    cari_landmarks, cari_paths = load_landmarks_from_single_path(cari_paths, cari_train_photo_landmarks_path)
    print('Loading face training landmarks done.')

    print('Loading test landmarks.')
    test_face_landmarks, face_test_paths = load_landmarks_from_single_path(face_test_paths, test_landmark_path)
    print('Loading test landmarks done.')

    print('Loading cari training landmarks done.')

    # training sample num
    face_train_num = len(train_face_landmarks)
    face_test_num = len(test_face_landmarks)
    cari_train_num = len(cari_landmarks)

    # convert to tensors, dim is N * (d*2).
    if use_web_cari:
        fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks) * cari_scale
    else:
        fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks) * face_scale

    cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks)
    test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks) * face_scale

    test_face_landmarks = test_face_landmarks * face_scale

    start = time.time()
    print(
        f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
    # perform WCT feature transformation
    wct_landmarks, test_wct_landmarks, face_mu, cari_mu = whiten_and_color_mu(fl_tensor, cl_tensor, test_fl_tensor)

    # record mean
    face_mean_landmarks_plot = np.ones((512, 512, 3), dtype=np.uint8) * 255
    face_mean_landmarks_plot = draw_key_points(face_mean_landmarks_plot, cari_mu)
    cv2.imwrite(str(record_output / 'fm.jpg'), face_mean_landmarks_plot)
    train_faces = []
    for p in face_paths:
        face = cv2.imread(str(p))
        train_faces.append(face)
    mean_face = np.stack(train_faces).mean(axis=0)
    cv2.imwrite(str(record_output / 'mean_face.jpg'), mean_face)

    # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
    train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
    # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    # pm_pairs = []

    if not warped_output.exists():
        warped_output.mkdir(parents=True, exist_ok=True)
    if not cmp_output.exists():
        cmp_output.mkdir(parents=True, exist_ok=True)

    pairs = []
    for idx, fp in enumerate(face_test_paths):
        pairs.append(
            (test_face_landmarks[idx], test_wct_landmarks[idx], fp, warped_output, cmp_output, scale, scale_to_old,
             draw_pts, face_scale, cari_scale,))
    # for idx, fp in enumerate(face_train_paths):
    #     pairs.append(
    #         (train_face_landmarks[idx], train_wct_landmarks[idx], fp, warped_output, cmp_output))

    # acceleration using multiprocessing
    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)


def warp_use_test(web_cari_path: Path, web_cari_test_path: Path, train_face_num, train_cari_num, device_id
                  , landmarks_num, warped_output, cmp_output, record_output, compress_face_distribution=False,
                  compress_cari_distribution=False, scale=-1, scale_to_old=False, fmt='.bmp', draw_pts=False,
                  cari_train_path=None,
                  train_face_scale=1, train_cari_scale=1, test_scale=1, rollback=False, sample_per_person=2):
    """
    :param web_cari_path:
    :param web_cari_test_path:
    :param train_face_num:
    :param train_cari_num:
    :param device_id:
    :param landmarks_num:
    :param warped_output:
    :param cmp_output:
    :param compress_face_distribution:
    :param compress_cari_distribution:
    :return:
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(landmarks_num)
    photo_paths = list(web_cari_photo_path.glob(f'*/P*{fmt}'))

    if cari_train_path is None:
        cari_paths = list(web_cari_photo_path.glob(f'*/C*{fmt}'))
    else:
        cari_train_photo_path = cari_train_path / 'img'
        cari_train_landmarks_path = cari_train_path / 'landmarks'
        cari_paths = list(cari_train_photo_path.glob(f'*'))
        if compress_cari_distribution:
            if train_cari_num <= len(cari_paths):
                cari_paths = sample(cari_paths, train_cari_num)
                print(cari_paths)

    # shuffles the datasets
    np.random.shuffle(photo_paths)
    np.random.shuffle(cari_paths)

    print(f'Photo num {len(photo_paths)}')
    print(f'Cari num {len(cari_paths)}')

    # loads face training landmarks

    if web_cari_test_path is None:
        print('Loading face training landmarks.')
        face_landmarks, face_paths = load_landmarks(photo_paths, web_cari_landmarks_path)
        print('Loading face training landmarks done.')
        face_train_paths, face_test_paths, train_face_landmarks, test_face_landmarks, = split_datasets(face_landmarks,
                                                                                                       face_paths,
                                                                                                       train_face_num)
    else:
        face_test_names = os.listdir(str(web_cari_test_path / 'img'))
        face_names = os.listdir(str(web_cari_path / 'img'))
        face_names = [n for n in face_names if n not in face_test_names]
        face_train_paths = []
        if compress_face_distribution:
            face_train_paths = sample_normal_distribution(face_names, sample_per_person, train_face_num, web_cari_path)
        else:
            for fn in face_names:
                face_train_paths.extend(list((web_cari_path / 'img' / fn).glob('P*')))

        # print(len(face_test_names))
        # supply_names = sample(face_names, half - len(face_test_names))
        # for spn in supply_names:
        #     shutil.copytree(web_cari_path / 'img' / spn, web_cari_test_path / 'img' / spn, dirs_exist_ok=True)
        #     shutil.copytree(web_cari_path / 'landmarks' / str(landmarks_num) / spn,
        #                     web_cari_test_path / 'landmarks' / str(landmarks_num) / spn, dirs_exist_ok=True)
        face_test_paths = list(web_cari_test_path.glob(f'*/*/P*{fmt}'))
        web_cari_test_landmarks_path = web_cari_test_path / 'landmarks' / str(landmarks_num)
        test_face_landmarks, face_test_paths = load_landmarks(face_test_paths, web_cari_test_landmarks_path)
        # fact_test_str_paths = [os.path.join(str(p).split('/')[-2], str(p).split('/')[-1]) for p in face_test_paths]
        # face_train_paths = [p for p in photo_paths if
        #                     os.path.join(str(p).split('/')[-2], str(p).split('/')[-1]) not in fact_test_str_paths]
        # if compress_face_distribution:
        #     face_train_paths = sample(face_train_paths, train_face_num) if len(
        #         face_train_paths) <= train_cari_num else face_train_paths
        train_face_landmarks, face_train_paths = load_landmarks(face_train_paths, web_cari_landmarks_path)

    # loads face test landmarks
    # print('Loading face test landmarks.')
    # test_face_landmarks = load_landmarks_from_web_cari(face_test_paths, web_cari_landmarks_path)
    # print('Loading face test done.')

    # loads cari training landmarks
    print('Loading cari training landmarks.')
    if cari_train_path is None:
        if compress_cari_distribution:
            face_test_names = os.listdir(str(web_cari_test_path / 'img'))
            face_names = os.listdir(str(web_cari_path / 'img'))
            face_names = [n for n in face_names if n not in face_test_names]
            # cari_paths = sample(cari_paths, train_cari_num)
            cari_paths = sample_normal_distribution(face_names, sample_per_person, train_cari_num, web_cari_path,
                                                    fmt='C*')
        cari_landmarks, cari_paths = load_landmarks(cari_paths, web_cari_landmarks_path)
    else:
        cari_landmarks, cari_paths = load_landmarks_from_single_path(cari_paths, cari_train_landmarks_path)

    with open(f'{record_output}/train_face_samples.txt', 'w') as fw:
        for path in face_train_paths:
            fw.write(str(path) + '\n')
    with open(f'{record_output}/train_cari_samples.txt', 'w') as fw:
        for path in cari_paths:
            fw.write(str(path) + '\n')
    # loads cari training landmarks
    print('Loading cari training landmarks done.')

    if rollback:
        tfs = f'{record_output}/train_face_samples.txt'
        if os.path.exists(tfs):
            with open(tfs, 'r') as f:
                face_train_paths = [Path(path.replace('\n', '')) for path in f.readlines()]
                train_face_landmarks, face_train_paths = load_landmarks(face_train_paths, web_cari_landmarks_path)
            with open(f'{record_output}/train_cari_samples.txt', 'r') as f:
                cari_paths = [Path(path.replace('\n', '')) for path in f.readlines()]
                cari_landmarks, cari_paths = load_landmarks(cari_paths, web_cari_landmarks_path)
    # training sample num
    face_train_num = len(train_face_landmarks)
    face_test_num = len(test_face_landmarks)
    cari_train_num = len(cari_landmarks)

    # convert to tensors, dim is N * (d*2).
    fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks) * train_face_scale
    cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks) * train_cari_scale
    test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks) * test_scale

    train_face_landmarks = train_face_landmarks * train_face_scale
    cari_landmarks = cari_landmarks * train_cari_scale
    test_face_landmarks = test_face_landmarks * test_scale
    start = time.time()
    print(
        f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
    # perform WCT feature transformation
    wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
    # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
    train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
    # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    # pm_pairs = []

    if not warped_output.exists():
        warped_output.mkdir(parents=True, exist_ok=True)
    if not cmp_output.exists():
        cmp_output.mkdir(parents=True, exist_ok=True)

    pairs = []
    for idx, fp in enumerate(face_test_paths):
        pairs.append(
            (test_face_landmarks[idx], test_wct_landmarks[idx], fp, warped_output, cmp_output, scale, scale_to_old,
             draw_pts, test_scale,))
    # for idx, fp in enumerate(face_train_paths):
    #     pairs.append(
    #         (train_face_landmarks[idx], train_wct_landmarks[idx], fp, warped_output, cmp_output))

    # acceleration using multiprocessing
    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)


def warp_single_test(web_cari_path: Path, train_photo_num, train_cari_num,
                     test_photo_num, test_cari_num, output_path,
                     fmt='.jpg', web_test_photo_path=None, web_test_cari_path=None):
    """
    单漫画形变迁移
    :param web_cari_path:
    :param train_photo_num:
    :param train_cari_num:
    :param test_photo_num:
    :param test_cari_num:
    :param output_path:
    :param fmt:
    :param web_test_photo_path:
    :param web_test_cari_path:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    Path(output_path).mkdir(exist_ok=True, parents=True)
    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(128)
    photo_paths = list(web_cari_photo_path.glob(f'*/P*{fmt}'))
    cari_paths = list(web_cari_photo_path.glob(f'*/C*{fmt}'))

    # shuffles the datasets
    np.random.shuffle(photo_paths)
    np.random.shuffle(cari_paths)

    train_photo_paths = sample(photo_paths, train_photo_num)
    train_cari_paths = sample(cari_paths, train_cari_num)

    if web_test_photo_path is not None:
        test_photo_paths = list(Path(web_test_photo_path).glob('P*'))
        # test_photo_paths = [web_test_photo_path]
    else:
        test_photo_paths = sample(photo_paths, test_photo_num)

    if web_test_cari_path is not None:
        # test_cari_paths = list(Path(web_test_cari_path).glob('*'))
        test_cari_paths = [web_test_cari_path]
    else:
        test_cari_paths = sample(cari_paths, test_cari_num)

    print(f'Photo num {len(train_photo_paths)}')
    print(f'Cari num {len(train_cari_paths)}')

    train_face_landmarks, train_face_paths = load_landmarks(train_photo_paths, web_cari_landmarks_path)
    train_cari_landmarks, train_cari_paths = load_landmarks(train_cari_paths, web_cari_landmarks_path)

    train_photo_num = len(train_face_paths)
    train_cari_num = len(train_cari_paths)

    train_photo_tensors = cvt_landmarks_distribution(device, train_photo_num, train_face_landmarks)
    train_cari_tensors = cvt_landmarks_distribution(device, train_cari_num, train_cari_landmarks)

    photo_mean = torch.mean(train_photo_tensors, 1)  # c x (h x w)
    cari_mean = torch.mean(train_cari_tensors, 1)

    for idx, photo_path in enumerate(test_photo_paths):
        test_face_landmark, test_face_path = load_landmarks([photo_path], web_cari_landmarks_path)
        test_cari_landmark, test_cari_path = load_landmarks([test_cari_paths[idx]], web_cari_landmarks_path)
        if not len(test_face_path) or not len(test_cari_path):
            continue

        test_photo_tensor = cvt_landmarks_distribution(device, 1, test_face_landmark)
        test_cari_tensor = cvt_landmarks_distribution(device, 1, test_cari_landmark)

        print(test_photo_tensor.size())
        print(test_cari_tensor.size())
        wct_landmark_1, test_wct_landmark_1 = whiten_and_color(test_photo_tensor, test_cari_tensor,
                                                               test_photo_tensor, use_mean=True)
        wct_landmark_2, test_wct_landmark_2 = whiten_and_color(test_photo_tensor, test_cari_tensor,
                                                               test_photo_tensor, photo_mean, cari_mean)
        face = cv2.imread(str(photo_path))
        cari = cv2.imread(str(test_cari_path[0]))

        test_face_landmark = test_face_landmark.reshape((1, 128, 2))
        test_wct_landmark_1 = test_wct_landmark_1.permute(1, 0).view(1, 128, 2).long().cpu().numpy()
        test_wct_landmark_2 = test_wct_landmark_2.permute(1, 0).view(1, 128, 2).long().cpu().numpy()
        warped_1, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_1[0])
        warped_2, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_2[0])
        # warped, transform = warp_image(face, wct_landmark, face_landmark)
        # convert to uint8 format image

        cari = draw_key_points(cari, test_cari_landmark[0])
        warped_1 = (warped_1 * 255).astype(np.uint8)
        warped_2 = (warped_2 * 255).astype(np.uint8)
        output = np.hstack([face, cari, warped_1, warped_2])
        cv2.imwrite(f'{output_path}/{idx}.png', output)


def warp_interplation(web_cari_path: Path, train_photo_num, train_cari_num,
                      test_photo_num, test_cari_num, output_path,
                      fmt='.jpg', web_test_photo_path=None, web_test_cari_path=None):
    """
    形变线性插值实验
    :param web_cari_path:
    :param train_photo_num:
    :param train_cari_num:
    :param test_photo_num:
    :param test_cari_num:
    :param output_path:
    :param fmt:
    :param web_test_photo_path:
    :param web_test_cari_path:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    # web_cari_path = Path('datasets/WebCari_512')
    # warp_interplation(web_cari_path, 10, 10, 100, 100, f'output/random/{generate_time_stamp()}')
    # 上面是调用这个函数时，传入的参数
    Path(output_path).mkdir(exist_ok=True, parents=True)
    # use default file layout
    web_cari_photo_path = web_cari_path / 'img'
    web_cari_landmarks_path = web_cari_path / 'landmarks' / str(128)
    photo_paths = list(web_cari_photo_path.glob(f'*/P*{fmt}'))
    # 列出P*目录下所有的jpg格式文件，并且返回参数为一个列表list
    cari_paths = list(web_cari_photo_path.glob(f'*/C*{fmt}'))

    # shuffles the datasets
    np.random.shuffle(photo_paths)
    np.random.shuffle(cari_paths)

    train_photo_paths = sample(photo_paths, train_photo_num)
    # 从上面返回的list中随机抽取train_photo_num（10）个jpg图片
    train_cari_paths = sample(cari_paths, train_cari_num)

    if web_test_photo_path is not None:
        test_photo_paths = list(Path(web_test_photo_path).glob('P*'))
        # test_photo_paths = [web_test_photo_path]
    else:
        test_photo_paths = sample(photo_paths, test_photo_num)

    if web_test_cari_path is not None:
        # test_cari_paths = list(Path(web_test_cari_path).glob('*'))
        test_cari_paths = [web_test_cari_path]
    else:
        test_cari_paths = sample(cari_paths, test_cari_num)
        # 默认的情况就是test的图片采样自photo图片

    print(f'Photo num {len(train_photo_paths)}')
    print(f'Cari num {len(train_cari_paths)}')

    train_face_landmarks, train_face_paths = load_landmarks(train_photo_paths, web_cari_landmarks_path)
    # 返回的第一个值为每一个photo（C开头的jpg）对应的landmark构成的list，第二个值就是train_photo_paths
    train_cari_landmarks, train_cari_paths = load_landmarks(train_cari_paths, web_cari_landmarks_path)
    # 返回的第一个值为每一个漫画（P开头的jpg）对应的landmark构成的list，第二个值就是train_photo_paths
    train_photo_num = len(train_face_paths)
    train_cari_num = len(train_cari_paths)

    train_photo_tensors = cvt_landmarks_distribution(device, train_photo_num, train_face_landmarks)
    train_cari_tensors = cvt_landmarks_distribution(device, train_cari_num, train_cari_landmarks)

    photo_mean = torch.mean(train_photo_tensors, 1)  # c x (hw)
    cari_mean = torch.mean(train_cari_tensors, 1)

    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for idx, photo_path in enumerate(test_photo_paths):
        if idx > len(test_photo_paths) - 1:
            break
        test_face_landmark, test_face_path = load_landmarks([test_photo_paths[idx]], web_cari_landmarks_path)
        test_cari_landmark, test_cari_path = load_landmarks([test_cari_paths[idx]], web_cari_landmarks_path)
        test_cari_landmark_2, test_cari_path_2 = load_landmarks([test_cari_paths[idx + 1]], web_cari_landmarks_path)
        if not len(test_face_path) or not len(test_cari_path) or not len(test_cari_path_2):
            continue

        test_photo_tensor = cvt_landmarks_distribution(device, 1, test_face_landmark)
        test_cari_tensor = cvt_landmarks_distribution(device, 1, test_cari_landmark)
        test_cari_tensor_2 = cvt_landmarks_distribution(device, 1, test_cari_landmark_2)

        print(test_photo_tensor.size())
        print(test_cari_tensor.size())
        wct_landmark_1, test_wct_landmark_1 = whiten_and_color(test_photo_tensor, test_cari_tensor,
                                                               test_photo_tensor, use_mean=True)
        wct_landmark_2, test_wct_landmark_2 = whiten_and_color(test_photo_tensor, test_cari_tensor_2,
                                                               test_photo_tensor, use_mean=True)
        face = cv2.imread(str(photo_path))
        cari_1 = cv2.imread(str(test_cari_path[0]))
        cari_2 = cv2.imread(str(test_cari_path_2[0]))

        test_face_landmark = test_face_landmark.reshape((1, 128, 2))
        test_wct_landmark_1 = test_wct_landmark_1.permute(1, 0).view(1, 128, 2).long().cpu().numpy()
        test_wct_landmark_2 = test_wct_landmark_2.permute(1, 0).view(1, 128, 2).long().cpu().numpy()

        warped_1, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_1[0])
        warped_2, transform = warp_image(face, test_face_landmark[0], test_wct_landmark_2[0])
        warped_1 = (warped_1 * 255).astype(np.uint8)
        warped_2 = (warped_2 * 255).astype(np.uint8)
        output = [face, cari_1, warped_1]

        for w in weights:
            test_wct_landmark = (1 - w) * test_wct_landmark_1 + w * test_wct_landmark_2
            warped, transform = warp_image(face, test_face_landmark[0], test_wct_landmark[0])
            warped = (warped * 255).astype(np.uint8)
            output.append(warped)
        output.append(warped_2)
        output.append(cari_2)
        output = np.hstack(output)
        cv2.imwrite(f'{output_path}/{idx}.png', output)


def sample_normal_distribution(names, sample_per_person, k, web_cari_path, fmt='P*'):
    paths = []
    photos_of_person = {}
    for fn in names:
        photos_of_person[fn] = list((web_cari_path / 'img' / fn).glob(fmt))
    while len(paths) < k:
        for fn in names:
            if sample_per_person <= len(photos_of_person[fn]):
                current_samples = sample(photos_of_person[fn], sample_per_person)
                paths.extend(current_samples)
                photos_of_person[fn] = [p for p in photos_of_person[fn] if p not in current_samples]
            else:
                paths.extend(photos_of_person[fn])
            if len(paths) >= k:
                break
    return paths


def split_datasets(face_landmarks, face_paths, split):
    face_train_paths = face_paths[:-split]
    face_test_paths = face_paths[-split:]
    train_face_landmarks = face_landmarks[:-split]
    test_face_landmarks = face_landmarks[-split:]
    return face_train_paths, face_test_paths, train_face_landmarks, test_face_landmarks


def warp_articst_faces(cari_train_path: Path, cari_test_path: Path, articts_photo_path: Path, train_face_num,
                       articts_cari_landmarks_path: Path, device_id
                       , landmarks_num, warped_output, cmp_output, articts_scale=2):
    """
    :param cari_train_path: 
    :param cari_test_path: 
    :param articts_photo_path: 
    :param device_id: 
    :param landmarks_num: 
    :param warped_output: 
    :param cmp_output: 
    :return: 
    """"""
    """
    device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

    # use default file layout
    web_cari_photo_path = cari_train_path / 'img'
    web_test_photo_path = cari_test_path / 'img'

    # landmarks path
    web_cari_train_landmarks_path = cari_train_path / 'landmarks' / str(landmarks_num)
    web_cari_test_landmarks_path = cari_test_path / 'landmarks' / str(landmarks_num)

    # cari_train_photo_path = cari_train_path / 'img'
    # cari_train_landmarks_path = cari_train_path / 'landmarks'

    train_face_paths = list(web_cari_photo_path.glob('*/P*'))
    face_test_paths = list(web_test_photo_path.glob('*/P*'))

    train_face_paths = sample(train_face_paths, train_face_num)
    # shuffles the datasets
    np.random.shuffle(train_face_paths)
    # loads face training landmarks
    print('Loading face training landmarks.')
    train_face_landmarks, train_face_paths = load_landmarks(train_face_paths, web_cari_train_landmarks_path)
    print('Loading face training landmarks done.')

    cari_paths = list(articts_photo_path.glob('*.png'))

    print('Loading face test landmarks.')
    test_face_landmarks, face_test_paths = load_landmarks(face_test_paths, web_cari_test_landmarks_path)
    print('Loading face test landmarks done.')

    # face_test_paths = list(photo_test_path.glob(f'*/P*{fmt}'))

    # loads face test landmarks
    # print('Loading face test landmarks.')
    # test_face_landmarks = load_landmarks_from_web_cari(face_test_paths, web_cari_landmarks_path)
    # print('Loading face test done.')

    # loads cari training landmarks
    print('Loading cari training landmarks.')
    cari_landmarks, cari_paths = load_landmarks_2(cari_paths, articts_cari_landmarks_path)
    # loads cari training landmarks
    print('Loading cari training landmarks done.')

    # training sample num
    face_train_num = len(train_face_landmarks)
    face_test_num = len(test_face_landmarks)
    cari_train_num = len(cari_landmarks)

    # convert to tensors, dim is N * (d*2).
    fl_tensor = cvt_landmarks_distribution(device, face_train_num, train_face_landmarks)
    cl_tensor = cvt_landmarks_distribution(device, cari_train_num, cari_landmarks) * articts_scale
    test_fl_tensor = cvt_landmarks_distribution(device, face_test_num, test_face_landmarks)

    start = time.time()
    print(
        f'Processing whiting and color operation: Total content features:[{face_train_num}]/ style feature: [{cari_train_num}]')
    # perform WCT feature transformation
    wct_landmarks, test_wct_landmarks = whiten_and_color(fl_tensor, cl_tensor, test_fl_tensor)
    # test_wct_landmarks = whiten_and_color(test_fl_tensor, cl_tensor)
    print('Whiting and Color Done after [{}] seconds.'.format(time.time() - start))

    # train_face_landmarks = train_face_landmarks.reshape((face_train_num, landmarks_num, 2))
    # train_wct_landmarks = wct_landmarks.permute(1, 0).view(face_train_num, landmarks_num, 2).long().cpu().numpy()
    # wct_landmarks = wct_landmarks.reshape((test_wct_landmarks, landmarks_num, 2))
    test_face_landmarks = test_face_landmarks.reshape((face_test_num, landmarks_num, 2))
    test_wct_landmarks = test_wct_landmarks.permute(1, 0).view(face_test_num, landmarks_num, 2).long().cpu().numpy()
    print('Face landmarks size: [{}]'.format(test_face_landmarks.shape))
    print('WCT landmarks size: [{}]'.format(test_wct_landmarks.shape))
    # pm_pairs = []

    pairs = []
    for idx, fp in enumerate(face_test_paths):
        pairs.append(
            (test_face_landmarks[idx], test_wct_landmarks[idx], fp, warped_output, cmp_output,))
    # for idx, fp in enumerate(face_train_paths):
    #     pairs.append(
    #         (train_face_landmarks[idx], train_wct_landmarks[idx], fp, warped_output, cmp_output))

    # acceleration using multiprocessing

    with mp.Pool(mp.cpu_count() - 1) as p:
        p.starmap(wct_task, pairs)


def cvt_landmarks_distribution(device, dim, landmarks):
    """
    convert N * landmarks_num * 2 to (landmarks_num * 2) * N
    :param device:
    :param dim:
    :param landmarks:
    :param scale:
    :return:
    """
    distribution = torch.from_numpy(landmarks).view(dim, -1).permute(1, 0).double().to(device)
    # 即将原来dim个图片的landmarks展开成论文中提到的dxnp（dxnc）大小的tensor
    return distribution


def load_landmarks(paths, landmarks_path):
    # path是挑选出来的十个photo图片，landmarks_path
    # paths = datasets / WebCari_512 / img / * / P * {.jpg}
    # landmarks_path = datasets / WebCari_512 / landmarks / 128
    landmarks = []
    filter_paths = []
    for fp in paths:
        print('Loading landmarks [{}]...'.format(fp))
        # {}代表format中的值，{0}代表第一个
        # {}代表format中的值，{0}代表第一个
        # print(' {}	{}aa{}'.format(1, 2, 3))
        # 1   2aa3
        # print(' {2}	{2}aa{2}'.format(1, 2, 3))
        #  3  3aa3
        face_name = os.path.basename(str(fp))
        # os.path.basename返回path最后的文件名,即*.jpg
        index = os.path.splitext(face_name)[0]
        # os.path.splitext分离文件名与扩展名,最后[0]即返回文件名，而丢掉扩展名.jpg
        pdir = os.path.basename(os.path.dirname(str(fp)))
        # os.path.dirname去掉最后文件名，返回目录
        landmark_path = landmarks_path / pdir / 'landmarks' / (str(index) + '.txt')
        # 上面即得到每一个图片对应的landmark坐标txt文件
        if landmark_path.exists():
            landmarks.append(
                np.loadtxt(str(landmark_path)))
            # np.loadtxt的功能是读入数据文件，这里的数据文件要求每一行数据的格式相同
            filter_paths.append(fp)
    return np.array(landmarks), filter_paths


def load_landmarks_2(paths, landmarks_path):
    landmarks = []
    filter_paths = []
    for fp in paths:
        print('Loading landmarks [{}]...'.format(fp))
        face_name = os.path.basename(str(fp))
        index = os.path.splitext(face_name)[0]
        pdir = os.path.basename(os.path.dirname(str(fp)))
        landmark_path = landmarks_path / 'landmarks' / (str(index) + '.txt')
        if landmark_path.exists():
            landmarks.append(
                np.loadtxt(str(landmark_path)))
            filter_paths.append(fp)
    return np.array(landmarks), filter_paths


def load_landmarks_from_single_path(paths, landmarks_path):
    landmarks = []
    filter_paths = []
    for fp in paths:
        print('Loading landmarks [{}]...'.format(fp))
        face_name = os.path.basename(str(fp))
        index = os.path.splitext(face_name)[0]
        landmark_path = landmarks_path / (str(index) + '.txt')
        if landmark_path.exists():
            landmarks.append(
                np.loadtxt(str(landmark_path)))
            filter_paths.append(fp)
            # img = cv2.imread(str(fp))
            # img1 = draw_key_points(img, np.loadtxt(str(landmark_path)))
            # p = os.path.join(os.path.dirname(str(fp)), index + '_pts.jpg')
            # cv2.imwrite(p, img1)
            # img2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            # img2 = draw_key_points(img2, np.loadtxt(str(landmark_path)) * 0.5)
            # p = os.path.join(os.path.dirname(str(fp)), index + '_pts2.jpg')
            # cv2.imwrite(p, img2)

    return np.array(landmarks), filter_paths


def wct_task(face_landmark, wct_landmark, fp, warped_output: Path, cmp_output: Path, scale=-1, scale_to_old=False,
             draw_ktps=False, face_scale=-1, cari_scale=-1):
    face_name = os.path.basename(os.path.dirname(fp)).replace(' ', '_')
    index = os.path.splitext(os.path.basename(fp))[0]

    warped_output = warped_output / face_name
    cmp_output = cmp_output / face_name
    # construct warp image path
    if scale != -1:
        displacement = wct_landmark - face_landmark
        if scale_to_old:
            wct_landmark = face_landmark + (displacement * scale).astype(face_landmark.dtype)
        else:
            wct_landmark = wct_landmark + (displacement * scale).astype(wct_landmark.dtype)

    if not warped_output.exists():
        warped_output.mkdir(parents=True, exist_ok=True)
    if not cmp_output.exists():
        cmp_output.mkdir(parents=True, exist_ok=True)

    # photo_kpts_path = os.path.join(str(warped_output), index + '_1.png')
    # warped_path = os.path.join(str(warped_output), index + '.png')
    # cmp_save_path = os.path.join(str(cmp_output), index + '.png')
    warped_path = str(warped_output / (index + '.png'))
    cmp_save_path = str(cmp_output / (index + '.png'))
    # load photo image
    face = cv2.imread(str(fp))
    if face_scale != -1:
        face = cv2.resize(face, (0, 0), fx=face_scale, fy=face_scale)
    if face is None:
        return
    # warp image
    warped, transform = warp_image(face, face_landmark, wct_landmark)
    # warped, transform = warp_image(face, wct_landmark, face_landmark)
    # convert to uint8 format image
    warped = (warped * 255).astype(np.uint8)
    # output warped image into warped_path
    print('Processed warping done based WCT landmarks: [{}]'.format(fp))
    # draw keypoints in image
    warped_kpts = draw_key_points(warped, wct_landmark)
    photo_kpts = draw_key_points(face, face_landmark)
    # stack original image and warpe image with keypoints
    # cmp = np.hstack((photo_kpts, warped_kpts))

    # cmp = np.hstack((face, warped))
    # cv2.imwrite(str(cmp_save_path), cmp)
    cv2.imwrite(warped_path, warped)
    if draw_ktps:
        photo_kpts_path = str(cmp_output / (index + '_pkpts.png'))
        warp_kpts_path = str(cmp_output / (index + '_kpts.png'))
        cv2.imwrite(str(photo_kpts_path), photo_kpts)
        cv2.imwrite(str(warp_kpts_path), warped_kpts)


def generate_time_stamp(fmt='%m%d%H%M'):
    return time.strftime(fmt, time.localtime(time.time()))


def cat_cmp(paths, original, output: Path, fmt='.jpg'):
    print(paths[0].exists())
    names = os.listdir(str(paths[0]))
    i = 0
    if not output.exists():
        output.mkdir(exist_ok=True, parents=True)
    for n in names:
        warps = os.listdir(str(paths[0] / n))
        plots = []
        for w in warps:
            op = original / n.replace('_', ' ') / (os.path.splitext(w)[0] + fmt)
            oi = cv2.imread(str(original / n.replace('_', ' ') / (os.path.splitext(w)[0] + fmt)))
            # oi = cv2.resize(oi, (0, 0), fx=2, fy=2)
            h_plots = [oi,
                       cv2.imread(str(paths[0] / n / w))]
            for rp in paths[1:]:
                h_plots.append(cv2.imread(str(rp / n / w)))
            h_plots = np.hstack(h_plots)
            plots.append(h_plots)
            if (i + 1) % 10 == 0:
                plots = np.vstack(plots)
                output_name = output / n
                output_name.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(output_name / f'{str(i)}.png'), plots)
                plots = []
            i += 1


def cat_art(path, original, person_names, arts: Path, output: Path, fmt='.jpg'):
    # names = os.listdir(str(paths[0]))
    i = 0
    if not output.exists():
        output.mkdir(exist_ok=True, parents=True)
    art_names = [a for a in os.listdir(str(arts)) if not a.startswith('.')]
    person_names = [p.replace(' ', '_') for p in person_names if not p.startswith('.')]
    for n in person_names:
        print(f'Processing {n}')
        a_plots = [np.ones((512, 512, 3), dtype=np.uint8) * 255]
        for an in art_names:
            art_path = sample(list((arts / an).glob('*.png')), 1)
            art_sample = cv2.resize(cv2.imread(str(art_path[0])), (0, 0), fx=2, fy=2)
            a_plots.append(art_sample)
        a_plots = np.hstack(a_plots)
        plots = [a_plots]
        nps = path / art_names[0] / 'warp' / n
        if not nps.exists():
            continue
        photo_names = [p for p in os.listdir(nps) if not p.startswith('.')]
        for w in photo_names:
            oi = cv2.imread(str(original / n.replace('_', ' ') / (os.path.splitext(w)[0] + fmt)))
            h_plots = [oi]
            for an in art_names:
                h_plots.append(cv2.imread(str(path / an / 'warp' / n / w)))
            h_plots = np.hstack(h_plots)
            plots.append(h_plots)
            if (i + 1) % 10 == 0:
                plots = np.vstack(plots)
                output_name = output / n
                output_name.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(output_name / f'{str(i)}.png'), plots)
                plots = [a_plots]
            i += 1


def ablation():
    web_cari_path = Path('datasets/WebCari-bmp')
    web_test_path = Path('datasets/WebCariTest-bmp')
    # web_cari_test = Path('datasets/WebCariTrain/img/Audrey')
    # cari_train_path = Path('datasets/Articst-face')
    # articts_photo_path = Path('datasets/Articst-faces/img/Amedeo_Modigliani')
    # articts_landmarks_path = Path('datasets/Articst-faces/landmarks/Amedeo_Modigliani')
    # articts_photo_path = Path('datasets/Articst-faces/img/Moise_Kisling')
    # articts_landmarks_path = Path('datasets/Articst-faces/landmarks/Moise_Kisling')
    train_face_num = 500
    train_cari_num = 500
    compress_face_distribution = True
    compress_cari_distribution = True
    scale = 2
    scale_to_old = True
    device_id = 0
    landmarks_num = 128
    time_stamp = generate_time_stamp()
    # time_stamp = '04151701'
    # time_stamp = '04051141'
    warped_output = Path(f'output/ablation/{time_stamp}/warp')
    cmp_output = Path(f'output/ablation/{time_stamp}/cmp')
    record_output = Path(f'output/ablation/{time_stamp}/record')
    use_web_cari = False
    # warp_ablation(web_cari_path, [1, 5, 15, 20], 1, 0, landmarks_num, warped_output, cmp_output, record_output,
    #               random_sample_times=2, scale=scale, scale_to_old=scale_to_old, fmt='.bmp')
    warp_ablation2(web_cari_path, [1, 5, 15, 20], 1, 0, landmarks_num, warped_output, cmp_output, record_output,
                   random_sample_times=2, scale=scale, scale_to_old=scale_to_old, fmt='.bmp', train_cari_scale=2,
                   train_face_scale=2, test_scale=2)
    # warp_fixed(web_cari_path, 100, 1000, 0, landmarks_num, warped_output, cmp_output, record_output,
    #            random_sample_times=2, fmt='.bmp')
    # result_path = Path(f'/Users/luvletteru/Documents/GitHub/facepp-python-sdk/output/ablation/{time_stamp}')
    # result_path = Path(f'/Users/luvletteru/Documents/GitHub/facepp-python-sdk/output/ablation/{time_stamp}')
    # cat_from_two_times(result_path, web_cari_path / 'img', [1, 5, 15, 20])
    # warp_articst_faces(web_cari_path, web_test_path, articts_photo_path, articts_landmarks_path, 0, landmarks_num, warped_output,
    #                    cpm_output)


def random_warp():
    celeb_path = Path('datasets/CelebA')
    cari_train_path = Path('datasets/WebCari_104')
    web_cari_path = Path('datasets/WebCari_512')
    web_test_path = Path('datasets/WebCariTest_512')
    fmt = '.jpg'
    # fmt = '.bmp'
    # web_cari_path = Path('datasets/WebCari-bmp')
    # web_test_path = Path('datasets/WebCariTest-bmp')

    train_face_num = 1000
    train_cari_num = 2000
    compress_face_distribution = True
    compress_cari_distribution = True
    scale = 1
    scale_to_old = True
    train_face_scale = 1
    test_scale = 1
    train_cari_scale = 1
    device_id = 0
    landmarks_num = 128
    # train_cari_nums = [20, 40, 60, 80, 100, 100, 500, 1000, 2000]
    train_cari_nums = [1]
    # time_stamps = ['04181538', '04181539', '04181541', '04181542', '04181543']
    time_stamps = []
    for idx, train_cari_num in enumerate(train_cari_nums):
        if idx < len(time_stamps):
            time_stamp = time_stamps[idx]
        else:
            time_stamp = generate_time_stamp()
        # time_stamp = '04171006'
        if scale_to_old:
            warped_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}/warp')
            cmp_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}/cmp')
            record_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}/record')
            record_output.mkdir(exist_ok=True, parents=True)
        else:
            warped_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}_r/warp')
            cmp_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}_r/cmp')
            record_output = Path(f'output/random/{time_stamp}_{train_face_num}_{train_cari_num}_{scale}/record')
            record_output.mkdir(exist_ok=True, parents=True)

        warp_use_test(web_cari_path, web_test_path, train_face_num, train_cari_num, 0, landmarks_num, warped_output,
                      cmp_output, record_output,
                      compress_face_distribution=compress_face_distribution,
                      compress_cari_distribution=compress_cari_distribution, scale=scale, scale_to_old=scale_to_old,
                      cari_train_path=cari_train_path, train_face_scale=train_face_scale, test_scale=test_scale,
                      train_cari_scale=train_cari_scale, draw_pts=False,
                      fmt=fmt)
        time_stamps.append(time_stamp)

    # test_path = Path('datasets/CelebA_contents')
    # train_face_num = 500
    # train_cari_num = 20
    # compress_face_distribution = True
    # compress_cari_distribution = False
    # load_from_file = False
    # warp_use_train_and_test(celeb_path, test_path, cari_train_path, train_face_num, train_cari_num, 0, landmarks_num,
    #                         warped_output,
    #                         cmp_output, record_output,
    #                         compress_face_distribution=compress_face_distribution,
    #                         compress_cari_distribution=compress_cari_distribution, scale=scale,
    #                         scale_to_old=scale_to_old, draw_pts=True, face_scale=0.5, cari_scale=2, use_web_cari=False,
    #                         load_from_file=load_from_file)


def articts():
    celeb_path = Path('datasets/CelebA')
    web_cari_path = Path('datasets/WebCari_512')
    web_test_path = Path('datasets/WebCariTest_512')
    # at_names = ['Amedeo_Modigliani', 'Comics', 'Egon_Schiele', 'Fernand_Leger']
    # at_names = ['Amedeo_Modigliani']
    at_names = os.listdir(str('datasets/Articst-faces/img'))
    print(at_names)
    landmarks_num = 128
    train_face_num = 1000
    for name in at_names:
        warped_output = Path(f'output/articts/{name}/warp')
        cmp_output = Path(f'output/articts/{name}/cmp')
        cari_train_path = Path(f'datasets/Articst-faces/img/{name}')
        articts_landmarks_path = Path(f'datasets/Articst-faces/landmarks/{str(landmarks_num)}/{name}')
        # record_output = Path(f'output/articts/{name}/record')
        warp_articst_faces(web_cari_path, web_test_path, cari_train_path, train_face_num, articts_landmarks_path, 0,
                           landmarks_num,
                           warped_output,
                           cmp_output)


def single_shape_exp():
    web_cari_path = Path('datasets/WebCari_512')
    # warp_single_test(web_cari_path, 1000, 500, 100, 100, f'output/random/{generate_time_stamp()}')
    # warp_single_test(web_cari_path, 10, 10, 100, 100, f'output/random/{generate_time_stamp()}')
    warp_interplation(web_cari_path, 10, 10, 100, 100, f'output/random/{generate_time_stamp()}')


if __name__ == '__main__':
    # ablation()
    # random_warp()
    # articts()
    single_shape_exp()
    # cat_cmp([
    #     Path(f'output/random/04191450_1000_20_1/warp'),
    #     Path(f'output/random/04191501_1000_40_1/warp'),
    #     Path(f'output/random/04191508_1000_60_1/warp'),
    #     Path(f'output/random/04191514_1000_80_1/warp'),
    #     Path(f'output/random/04191628_1000_100_1/warp'),
    #     Path(f'output/random/04191521_1000_100_1/warp'),
    #     Path(f'output/random/04191530_1000_100_1/warp'),
    #     Path(f'output/random/04191541_1000_500_1/warp'),
    #     Path(f'output/random/04191556_1000_1000_1/warp'),
    #     Path(f'output/random/04191608_1000_2000_1/warp')],
    #     Path('datasets/WebCari_512/img'),
    #     Path(f'output/random_cmp/{generate_time_stamp()}'))
    # cat_art(Path('output/articts'), Path('datasets/WebCari_512/img'), os.listdir('datasets/WebCariTest_512/img'),
    #         Path('datasets/Articst-faces/img'),
    #         Path(f'output/art_cmp/{generate_time_stamp()}'))
