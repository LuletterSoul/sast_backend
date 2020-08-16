import os
import time

from flask import request
from flask_socketio import (
    SocketIO,
    emit, join_room, leave_room
)

from config import Config

import logging

logger = logging.getLogger('gunicorn.error')


def create_socket():
    return SocketIO(async_mode='eventlet', message_queue=Config.REDIS_SOCKET_URL, cors_allowed_origins="*")


# def create_app_socket(app):
#     return SocketIO(app=app, async_mode='eventlet', message_queue=Config.REDIS_SOCKET_URL, cors_allowed_origins="*")


socketio = SocketIO()


@socketio.on('connect')
def connect():
    sid = request.sid
    join_room(sid)
    # print(f'Socket connection created with ')
    print(f'Socket connection created with {sid}')
    emit('onConnected', {'sid': sid}, room=sid)


@socketio.on('disconnect')
def disconnect():
    sid = request.sid
    leave_room(sid)
    print(f'Socket disconnected from {sid}')


@socketio.on_error()
def error():
    sid = request.sid
    print(f'Socket error from {sid}')


def synthesis_complete(body):
    """
    notify frontend that a stylization image is obtainable.
    :param body: {
            'content_id': content_id,
            'style_id': style_id,
            'stylization_id': stylization_id,
        }
    :return:
    """
    print(f'notify fronted synthesis completely with body: {body}')
    socket_connection = create_socket()
    socket_connection.emit('onSynthesisCompleted', body, broadcast=True, room=body['sid'])
    # socketio.emit('onSynthesisCompleted', body)


def synthesis_failed(body):
    """
    notify frontend that a stylization image is obtainable.
    :param body: {
            'content_id': content_id,
            'style_id': style_id,
            'stylization_id': stylization_id,
        }
    :return:
    """
    print(f'notify fronted synthesis failed with body: {body}')
    socket_connection = create_socket()
    socket_connection.emit('onSynthesisFailed', body, room=body['sid'])
    # socketio.emit('onSynthesisFailed', body, broadcast=True)


def synthesising(body):
    """
    notify frontend with current synthesis progress.
    The frontend mustn't fetching stylization image from backend if the image's status is 'SYNTHESISING'
    :param body: {
        'content_id': content_id,
        'style_id': style_id,
        'stylization_id': stylization_id,
        'current_update_steps': 100,
        'current_cost_time': 200,
        'percent': 0.35, # 1 represent 'COMPLETE',otherwise it is 'SYNTHESISING',
        'total_time': 10,
        'total_update_steps': 10,
    }
    :return:
    """
    print(f'notify fronted synthesis with body: {body}')
    socket_connection = create_socket()
    socket_connection.emit('onSynthesising', body, room=body['sid'])
    # socketio.emit('onSynthesising', body, broadcast=True)


def mast_report(req, res_queue):
    start_time = time.time()
    c_basename = os.path.splitext(req['content_img_id'])[0]
    s_basename = os.path.splitext(req['style_img_id'])[0]
    stylization_id = f'{c_basename}_{s_basename}.png'
    req_id = req['req_id']
    sid = req['sid']

    while True:
        # try:
        if not res_queue.empty():
            res_msg = res_queue.get()
            copy_res = res_msg.copy()
            print(f'Request msg {res_msg}')
            if req_id == res_msg['req_id']:
                body = {
                    'sid': sid,
                    'req_id': req_id,
                    'content_id': res_msg['content_img_id'],
                    'style_id': res_msg['style_img_id'],
                    'stylization_id': res_msg['stylized_img_id'],
                    'timestamp': time.time()
                }
                if res_msg['status'] == 'success':
                    synthesis_complete(body)
                else:
                    synthesis_failed(body)
                break
            else:
                res_queue.put(copy_res)
        else:
            time.sleep(0.5)
            cost_time = time.time() - start_time
            body = {
                'sid': sid,
                'req_id': req_id,
                'content_id': req['content_img_id'],
                'style_id': req['style_img_id'],
                'stylization_id': stylization_id,
                'current_update_steps': -1,
                'current_cost_time': cost_time,
                'percent': round(cost_time / Config.MAST_TOTAL_TIME * 100, 1),
                # 1 represent 'COMPLETE',otherwise it is 'SYNTHESISING',
                'total_time': Config.MAST_TOTAL_TIME,
                'total_update_steps': -1,
            }
            synthesising(body)
        print(f'MASK report thread exit from request id {req_id}.')
    # except Exception as e:
    #     traceback.print_exc()
