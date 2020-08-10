import os
import math
import linecache
from PIL import Image
import matplotlib.pyplot as plt
import copy
import threading
import numpy as np
import cv2
from utils.wct import whiten_and_color, warp_image, draw_key_points
from pathlib import Path

num_of_points = 17
time_enlarge = 1.4
resize_w = 512
resize_h = 512
path_data = 'datasets/WebCaricature'
path_output = f'datasets/WebCaricature_align_{time_enlarge}_512_jpg/'

landmarks_data = 'datasets/WebCari/landmarks/845'

path_images = os.path.join(path_data, 'OriginalImages')
path_points = os.path.join(path_data, 'FacialPoints')
path_output_images = os.path.join(path_output, 'image')
path_output_points = os.path.join(path_output, 'landmark')
path_output_landmarks = os.path.join(path_output, 'landmarks')
if not os.path.exists(path_output_landmarks):
    Path(path_output_landmarks).mkdir(exist_ok=True, parents=True)


def main():
    root = path_points
    image_list = list_all_images(root)
    start_threads(image_list)


def get_point_from_line(path, x):
    line = linecache.getline(path, x)
    pos = line.find(" ")
    x = line[:pos]
    y = line[pos:line.__len__() - 1]
    return float(x), float(y)


def get_points_from_txt(path):
    points = [[0] * 2 for i in range(num_of_points)]
    for i in range(num_of_points):
        x, y = get_point_from_line(path, i + 1)
        points[i][0] = x
        points[i][1] = y
    return points


def load_landmark(path):
    result = [[0] * 2 for i in range(num_of_points)]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    for i in range(17):
        result[i][0] = int(lines[i].split('\t')[0])
        result[i][1] = int(lines[i].split('\t')[1])
    return result


def get_rotate_angle(points):
    xl = (points[8][0] + points[9][0]) / 2
    yl = (points[8][1] + points[9][1]) / 2
    xr = (points[10][0] + points[11][0]) / 2
    yr = (points[10][1] + points[11][1]) / 2
    if xl == xr:
        if yr > yl:
            return 90
        elif yr < yl:
            return -90
        else:
            print(points)
            raise RuntimeError('x=x,y=y')
    tan_x = (yr - yl) / (xr - xl)
    x_rad = math.atan(tan_x)
    x_angle = (180 * x_rad) / math.pi
    return x_angle


def rotate_image(name, filename, angle):
    path_image = os.path.join(path_images, name, filename)
    image = Image.open(path_image)
    image_rotated = image.rotate(angle, Image.BILINEAR)
    w, h = image.size
    return image_rotated, w, h
    # path_output_image = os.path.join(path_output_images + name, 'r' + filename + '.jpg')
    # image_rotated.save(path_output_image, 'jpeg')


def calculate_new_point(x0, y0, angle, w, h):
    if angle == 0:
        return x0, y0
    angle *= -(math.pi / 180)
    x1 = x0 - w / 2
    y1 = y0 - h / 2
    r_square = x1 * x1 + y1 * y1
    if x1 == 0:
        tanx = -(1 / math.tan(angle))
    else:
        if y1 * math.tan(angle) == x1:
            x2 = 0
            y2 = math.sqrt(r_square)
            return x2, y2
        elif 1 - math.tan(angle) * (y1 / x1) == 0:
            x2 = 0
            y2 = math.sqrt(r_square)
            return x2, y2
        else:
            tanx = (y1 / x1 + math.tan(angle)) / (1 - math.tan(angle) * (y1 / x1))
    '''   
    if x1 != 0:
        if y1 * math.tan(angle) = x1
        temp1 = math.tan(angle)
        temp2 = y1 / x1
        tanx = (y1 / x1 + math.tan(angle)) / (1 - math.tan(angle) * (y1 / x1))
    else:
        tanx = -(1 / math.tan(angle))
    '''
    x2_square = r_square / (1 + tanx * tanx)
    x2_1 = math.sqrt(x2_square)
    y2_1 = x2_1 * tanx
    x2_2 = -x2_1
    y2_2 = -y2_1
    d1_square = (x1 - x2_1) * (x1 - x2_1) + (y1 - y2_1) * (y1 - y2_1)
    d2_square = (x1 - x2_2) * (x1 - x2_2) + (y1 - y2_2) * (y1 - y2_2)
    if d1_square < d2_square:
        x2 = x2_1
        y2 = y2_1
    else:
        x2 = x2_2
        y2 = y2_2
    x2 += w / 2
    y2 += h / 2
    return x2, y2


def calculate_new_points(points, angle, w, h):
    if angle == 0:
        return points
    else:
        result = [[0] * 2 for i in range(num_of_points)]
        for i in range(num_of_points):
            x0 = points[i][0]
            y0 = points[i][1]
            x2, y2 = calculate_new_point(x0, y0, angle, w, h)
            result[i][0] = x2
            result[i][1] = y2
        return result


def calculate_boundingbox(points):
    max_list = []
    min_list = []
    for j in range(len(points[0])):
        list = []
        for i in range(len(points)):
            list.append(points[i][j])
        max_list.append(max(list))
        min_list.append(min(list))
    x_max = max_list[0]
    y_max = max_list[1]
    x_min = min_list[0]
    y_min = min_list[1]

    # delta_x = x_max - x_min
    # delta_y = y_max - y_min
    # length = abs(delta_x - delta_y) / 2
    # if delta_x > delta_y:
    #     y_min -= length
    #     y_max += length
    # else:
    #     x_min -= length
    #     x_max += length

    return x_max, x_min, y_max, y_min


def calculate_boundingbox1(points):
    max_list = []
    min_list = []
    for j in range(len(points[0])):
        list = []
        for i in range(len(points)):
            list.append(points[i][j])
        max_list.append(max(list))
        min_list.append(min(list))
    x_max = max_list[0]
    y_max = max_list[1]
    x_min = min_list[0]
    y_min = min_list[1]

    # delta_x = x_max - x_min
    # delta_y = y_max - y_min
    # length = abs(delta_x - delta_y) / 2
    # if delta_x > delta_y:
    #     y_min -= length
    #     y_max += length
    # else:
    #     x_min -= length
    #     x_max += length

    return x_max, x_min, y_max, y_min


def calculate_boundingbox2(points):
    max_list = []
    min_list = []
    for j in range(len(points[0])):
        list = []
        for i in range(len(points)):
            list.append(points[i][j])
        max_list.append(max(list))
        min_list.append(min(list))
    x_max = max_list[0]
    y_max = max_list[1]
    x_min = min_list[0]
    y_min = min_list[1]

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    length = abs(delta_x - delta_y) / 2
    dx = 0
    dy = 0
    ny_min = y_min
    ny_max = y_max

    nx_min = x_min
    nx_max = x_max
    if delta_x > delta_y:
        ny_min -= length
        ny_max += length
        dy = length
    else:
        nx_min -= length
        nx_max += length
        dx = length

    return x_max, x_min, y_max, y_min, nx_max, nx_min, ny_max, ny_min, dx, dy


def enlarge(x_max, x_min, y_max, y_min, time_enlarge, w, h):
    # x_max += (time_enlarge - 1) * (x_max - x_min) / 2
    # y_max += (time_enlarge - 1) * (y_max - y_min) / 2
    # x_min -= (time_enlarge - 1) * (x_max - x_min) / 2
    # y_min -= (time_enlarge - 1) * (y_max - y_min) / 2
    nx_max = x_max + (time_enlarge - 1) * (x_max - x_min) / 2
    ny_max = y_max + (time_enlarge - 1) * (y_max - y_min) / 2
    nx_min = x_min - (time_enlarge - 1) * (x_max - x_min) / 2
    ny_min = y_min - (time_enlarge - 1) * (y_max - y_min) / 2
    '''
    x_min = 0 if x_min < 0 else x_min
    y_min = 0 if y_min < 0 else y_min
    x_max = w if x_max > w else x_max
    y_max = h if y_max > h else y_max
    '''
    return nx_max, nx_min, ny_max, ny_min


def look(img, points):
    plt.clf()
    xs = [points[i][0] for i in range(17)]
    ys = [points[i][1] for i in range(17)]
    plt.imshow(img)
    plt.scatter(xs, ys, s=16)
    plt.show()


def update_landmark_cropped(landmark, x_min, y_min):
    result = copy.deepcopy(landmark)
    for i in range(17):
        result[i][0] = landmark[i][0] - x_min
        result[i][1] = landmark[i][1] - y_min
    return result


def update_landmark_enlarged(landmark, w, h, resize_w, resize_h):
    time_w = resize_w / w
    time_h = resize_h / h
    for i in range(17):
        landmark[i][0] = landmark[i][0] * time_w
        landmark[i][1] = landmark[i][1] * time_h
    return landmark


def save_landmark(landmark, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(17):
            f.write(str(int(landmark[i][0])) + '\t' + str(int(landmark[i][1])) + '\n')
    f.close()


def list_all_images(root):
    result = []
    for name in os.listdir(root):
        for file in os.listdir(os.path.join(root, name)):
            result.append(os.path.join(root, name, file))
    return result


def start_threads(image_list, n_threads=16):
    if n_threads > len(image_list):
        n_threads = len(image_list)
    n = int(math.ceil(len(image_list) / float(n_threads)))
    print('the thread num is {}'.format(n_threads))
    print('each thread images num is {}'.format(n))
    image_lists = [image_list[index:index + n] for index in range(0, len(image_list), n)]
    thread_list = {}
    for thread_id in range(n_threads):
        thread_list[thread_id] = MyThread(image_lists[thread_id], thread_id)
        thread_list[thread_id].start()

    for thread_id in range(n_threads):
        thread_list[thread_id].join()


class MyThread(threading.Thread):
    def __init__(self, image_list, thread_id):
        threading.Thread.__init__(self)
        self.image_list = image_list
        self.thread_id = thread_id

    def run(self):
        print('thread {} begin'.format(self.thread_id))

        image_len = len(self.image_list)
        print_interval = image_len // 100
        print_interval = print_interval if print_interval > 0 else 1

        for index, image_path in enumerate(self.image_list):
            name = image_path.split('/')[-2]
            filename = image_path.split('/')[-1][:-4]
            try:
                path_point_txt = image_path
                # 获取关键点
                points = get_points_from_txt(path_point_txt)
                # 计算旋转角
                angle = get_rotate_angle(points)
                # 旋转图片
                im = Image.open(os.path.join(path_images, name, filename + '.jpg'))

                lp = os.path.join(landmarks_data, name, 'landmarks', filename + '.txt')

                if not os.path.exists(lp):
                    continue
                # look(im, points)
                image_rotated, w, h = rotate_image(name, filename + '.jpg', angle)
                # 计算旋转后的关键点
                points_rotated = calculate_new_points(points, angle, w, h)
                # look(image_rotated, points_rotated)
                # 计算BoundingBox边界
                _, _, _, _, x_max, x_min, y_max, y_min, _, _ = calculate_boundingbox2(points_rotated)

                pts = np.loadtxt(lp)
                npts = pts.copy()
                opts = pts.copy()

                alpha1 = 0.2
                alpha2 = time_enlarge - 1

                x_e, x_s, y_e, y_s, x2, x1, y2, y1, dx, dy = calculate_boundingbox2(points_rotated)

                delta_x1 = (alpha1 + 1) * (x_e - x_s)
                delta_y1 = (alpha1 + 1) * (y_e - y_s)
                e_x1 = x_s - alpha1 * (x_e - x_s) / 2
                e_y1 = y_s - alpha1 * (y_e - y_s) / 2

                opts[:0] = (delta_x1 / resize_w) * pts[:0] + e_x1
                opts[:1] = (delta_y1 / resize_h) * pts[:1] + e_y1

                # delta_x2 = (alpha2 + 1) * (x_e - x_s + 2 * dx)
                # delta_y2 = (alpha2 + 1) * (y_e - y_s + 2 * dy)
                # e_x2 = (x_s - dx) - alpha2 * (x_e - x_s + 2 * dx) / 2
                # e_y2 = (y_s - dy) - alpha2 * (y_e - y_s + 2 * dy) / 2
                delta_x2 = (alpha2 + 1) * (x2 - x1)
                delta_y2 = (alpha2 + 1) * (y2 - y1)
                e_x2 = x1 - alpha2 * (x2 - x1) / 2
                e_y2 = y1 - alpha2 * (y2 - y1) / 2

                npts[:, 0] = pts[:, 0] * (delta_x1 / delta_x2) + (e_x1 - e_x2) * (resize_w / delta_x2)
                npts[:, 1] = pts[:, 1] * (delta_y1 / delta_y2) + (e_y1 - e_y2) * (resize_h / delta_y2)

                # 放大BoundingBox边界
                x_max, x_min, y_max, y_min = enlarge(x_max, x_min, y_max, y_min, time_enlarge, w, h)
                x_max1, x_min1, y_max1, y_min1 = enlarge(x_e, x_s, y_e, y_s, alpha1 + 1, w, h)
                x_max2, x_min2, y_max2, y_min2 = enlarge(x2, x1, y2, y1, alpha2 + 1, w, h)
                # print(image_rotated.size[0], image_rotated.size[1])
                # 裁剪BoundingBox
                image_cropped = image_rotated.crop((x_min, y_min, x_max, y_max))
                image_cropped_1 = image_rotated.crop((x_min1, y_min1, x_max1, y_max1))
                image_cropped_2 = image_rotated.crop((x_min2, y_min2, x_max2, y_max2))

                # delta_x1_hat = (x_e - x_s) + (alpha1 * (x_e - x_s) / 2) + (alpha1 * (x_e - x_s) / 2)
                # delta_x2_hat = (x2 - x1) + (alpha2 * (x2 - x1) / 2) + (alpha2 * (x2 - x1) / 2)
                # print(f'IZero ratio: {delta_x1_hat, delta_x2_hat}')
                # 计算裁剪后的关键点
                points_cropped = update_landmark_cropped(points_rotated, x_min, y_min)
                # look(image_cropped, points_cropped)
                # 调整尺寸
                image_result = image_cropped.resize((resize_w, resize_h), Image.BILINEAR)
                image_result1 = image_cropped_1.resize((resize_w, resize_h), Image.BILINEAR)
                image_result2 = image_cropped_2.resize((resize_w, resize_h), Image.BILINEAR)

                # cv_image_result = cv2.cvtColor(np.array(image_rotated), cv2.COLOR_RGB2BGR)
                cv_image_result = cv2.cvtColor(np.array(image_result), cv2.COLOR_RGB2BGR)
                cv_image_result_1 = cv2.cvtColor(np.array(image_result1), cv2.COLOR_RGB2BGR)
                cv_image_result_2 = cv2.cvtColor(np.array(image_result2), cv2.COLOR_RGB2BGR)

                # cv_image_result = draw_key_points(cv_image_result, opts)
                cv_image_result_1 = draw_key_points(cv_image_result_1, pts)
                cv_image_result_2 = draw_key_points(cv_image_result_2, npts)

                cat_image = np.hstack([cv_image_result_1, cv_image_result_2])

                w_cropped, h_cropped = image_cropped.size
                # 计算缩放后的关键点
                points_result = update_landmark_enlarged(points_cropped, w_cropped, h_cropped, resize_w, resize_h)
                # look(image_result, points_result)
                dir = os.path.join(path_output_points, name)
                txt_name = filename + '.txt'
                save_landmark(points_result, dir, txt_name)

                landmark_path = Path(os.path.join(path_output_landmarks, name, 'landmarks'))
                landmark_path.mkdir(exist_ok=True, parents=True)
                np.savetxt(os.path.join(str(landmark_path), txt_name), npts, fmt="%d")
                # 保存
                path_output_image = os.path.join(path_output_images, name)
                if not os.path.exists(path_output_image):
                    os.makedirs(path_output_image)
                # image_rotated.save(path_output_image + filename + '_r.jpg', 'jpeg')
                # image_result.save(os.path.join(path_output_image, filename + '.bmp'), 'bmp')
                # image_result.save(os.path.join(path_output_image, filename + '.jpg'), 'jpg')

                # cat_image = Image.fromarray(cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB))
                # img2 = Image.fromarray(cv2.cvtColor(cv_image_result_2, cv2.COLOR_BGR2RGB))
                # cat_image.save(os.path.join(path_output_image, filename + '_cat.bmp'), 'bmp')
                # img2.save(os.path.join(path_output_image, filename + f'_1.bmp'), 'bmp')
                cv2.imwrite(os.path.join(path_output_image, filename + '.jpg'), cv_image_result)

                # img = Image.fromarray(cv2.cvtColor(cv_image_result, cv2.COLOR_BGR2RGB))
                # img.save(os.path.join(path_output_image, filename + '_o.bmp'), 'bmp')
            except ZeroDivisionError:
                print(name)
                print(filename)
            if index % print_interval == 0 and index > 0:
                print('{}/{} in thread {} has been sloven'
                      .format(index, image_len, self.thread_id))

        print('thread {} end.'.format(self.thread_id))


if __name__ == '__main__':
    main()
