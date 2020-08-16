#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_workers.py
@time: 2020/8/15 13:12
@version 1.0
@desc:
"""
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
from config import Config
from mast.libs.MastConfig import MastConfig
from workers.stream import *
from workers.tasks import MastModel

BATCH_SIZE = 2

msg = [{
    'req_id': str(uuid.uuid1()),
    'content_id': '0.bmp',
    'style_id': '0.bmp',
    'content_mask': [],
    'style_mask': [],
    'width': 512,
    'height': 512
}]


def test_init_redis_workers():
    root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../mast')
    cfg = MastConfig(os.path.join(root, f'configs/MAST_Configs.yml'))
    content_dir = Config.CONTENT_DIRECTORY
    style_dir = Config.STYLE_DIRECTORY
    stylized_dir = Config.STYLIZATION_DIRECTORY
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    os.makedirs(stylized_dir, exist_ok=True)
    model_init_args = (root, cfg, content_dir, style_dir, stylized_dir,)
    destroy_event = mp.Event()
    batch_size = 4
    worker_num = 4
    thread = threading.Thread(target=run_redis_workers_forever, args=(MastModel
                                                                      , batch_size, 0.1, worker_num, (0, 1, 2, 3),
                                                                      "localhost:6379", 'm1',
                                                                      model_init_args, None, destroy_event,),
                              daemon=True)

    # thread1 = threading.Thread(target=run_redis_workers_forever, args=(
    #     MastModel, batch_size, 0.1, 4, (0, 1, 2, 3,), "localhost:6379", 'm1', model_init_args, None, destroy_event,),
    #                            daemon=True)
    # thread2 = threading.Thread(target=run_redis_workers_forever, args=(
    #     MastModel, batch_size, 0.1, worker_num, (0, 1, 2, 3), "localhost:6379", 'm2', model_init_args, None,
    #     destroy_event,),
    #                            daemon=True)

    # thread3 = threading.Thread(target=run_redis_workers_forever, args=(
    #     MastModel, batch_size, 0.1, 4, (0, 1, 2, 3,), "localhost:6379", 'm2', model_init_args, None, destroy_event,),
    #                            daemon=True)

    thread.start()
    # thread1.start()
    # thread2.start()
    # thread3.start()
    return thread, destroy_event
    # return [thread, None, None,]


def test_redis_streamer():
    # Spawn releases 4 gpu worker processes
    streamer = RedisStreamer(prefix='m1')
    single_predict = streamer.predict(msg)
    # assert single_predict == .single_output
    # batch_predict = streamer.predict(single_predict)
    print(f'rec from {single_predict}')


def test_multi_channel_streamer():
    batch_size = 256

    streamer_1 = RedisStreamer(prefix='m1')
    streamer_2 = RedisStreamer(prefix='m2')

    t_start = time.time()
    xs = []
    for i in range(batch_size):
        future = streamer_1.submit(msg)
        xs.append(future)
    # for i in range(batch_size):
    #     future = streamer_2.submit(msg)
    #     xs.append(future)
    for future in tqdm(xs):  # 先拿到所有future对象，再等待异步返回
        output = future.result()
    t_end = time.time()
    # streamer_1.destroy_workers()
    # streamer_2.destroy_workers()
    destroy_event.set()
    print(f'[streamed]image per second: [{round(2 * batch_size / (t_end - t_start), 2)}]', )
    time.sleep(10)


if __name__ == '__main__':
    t1, destroy_event = test_init_redis_workers()
    # test_redis_streamer()
    time.sleep(15)
    test_multi_channel_streamer()
    t1.join()
