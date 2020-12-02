#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: utils.py
@time: 2020/8/16 23:44
@version 1.0
@desc:
"""
import os
import time
from flask import send_file
from PIL import Image
import io


def get_prefix(image_name):
    """
    get image name's prefix
    :param image_name:
    :return:
    """
    return os.path.splitext(os.path.basename(image_name))[0]


def compose_prefix_id(img_name1, img_name2):
    """
    compose stylized image id according content id and style id.
    :param img_name1:
    :param img_name2:
    :return:
    """
    return f'{compose_prefix(img_name1, img_name2)}.png'.replace(' ', '_')


def compose_prefix(img_name1, img_name2):
    return f'{get_prefix(img_name1)}-{get_prefix(img_name2)}'


def construct_cast_msg(msg, stylization_id, current_update_steps, current_cost_time, total_steps, category):
    """
    construct synthesising message body, used by client end.
    :param msg:
    :param stylization_id:
    :param current_update_steps:
    :param current_cost_time:
    :param total_steps:
    :return:
    """
    return {
        'sid': msg['sid'],
        'req_id': msg['req_id'],
        'content_id': msg['content_id'],
        'style_id': msg['style_id'],
        'stylization_id': stylization_id,
        'current_update_steps': current_update_steps,
        'current_cost_time': current_cost_time,
        'percent': round(current_update_steps / total_steps * 100, 1),
        # 1 represent 'COMPLETE',otherwise it is 'SYNTHESISING',
        'total_time': -1,
        'total_update_steps': total_steps,
        'timestamp': time.time(),
        'category': category,
    }


def parse_devices(device_string):
    """
    device string format as '0,1,2,3...'
    :param device_string:
    :return:
    """
    return [int(d) for d in device_string.split(',')]


def is_photo(fmt):
    return fmt == '.png' or fmt == '.jpg' or fmt == '.bmp'


def is_video(fmt):
    return fmt == '.mp4' or fmt == 'mpeg'


def send_img(path, filename, width, height, as_attachment):
    pil_image = Image.open(path)

    if pil_image is None:
        return {'success': False}, 400

    # we need different size image by parameters passed from client end.
    # width = args.get('width')
    # height = args.get('height')

    if not width:
        width = pil_image.size[1]
    if not height:
        height = pil_image.size[0]

    pil_image.thumbnail((width, height), Image.ANTIALIAS)
    file_io = io.BytesIO()
    pil_image.save(file_io, "PNG")
    file_io.seek(0)

    # complete all business logic codes here including image resizing and image transmission !

    # image must be resized by previous width and height
    # and I/O pipe must be built for bytes transmission between backend and client end
    return send_file(file_io, attachment_filename=filename, as_attachment=as_attachment)
