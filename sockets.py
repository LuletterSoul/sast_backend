import time

from flask import session
from flask_socketio import (
    SocketIO,
    disconnect,
    join_room,
    leave_room,
    emit
)
from config import Config

import eventlet

eventlet.monkey_patch(thread=False)

import logging

logger = logging.getLogger('gunicorn.error')

socketio = SocketIO(cors_allowed_origins="*")


@socketio.on('connect')
def connect():
    print(f'Socket connection created with')
    emit('server_response', {
        'stylization_id': 'test.png',
        'update_steps': 100,
        'total': 200,
        'percent': 0.35
    }, broadcast=True)


@socketio.on('disconnect')
def disconnect():
    print(f'Socket disconnected.')
    emit('server_response', {
        'stylization_id': 'test.png',
        'update_steps': 100,
        'total': 200,
        'percent': 0.35
    }, broadcast=True)


@socketio.on('synthesis')
def begin_synthesis():
    print(f'begin synthesis')
    emit('server_response', {
        'stylization_id': 'test.png',
        'update_steps': 100,
        'total': 200,
        'percent': 0.35
    }, broadcast=True)
