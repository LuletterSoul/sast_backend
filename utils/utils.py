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
