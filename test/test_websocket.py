#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_websocket.py
@time: 2020/8/10 22:04
@version 1.0
@desc:
"""

from app import socketio, app


def socketio_test():
    flask_test_client = app.test_client()
    socketio_test_client = socketio.test_client(app, flask_test_client=flask_test_client)
    socketio_test_client.emit('synthesis')


if __name__ == '__main__':
    socketio_test()
