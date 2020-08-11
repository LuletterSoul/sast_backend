#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: __init__.py.py
@time: 2020/8/8 16:11
@version 1.0
@desc:
"""
from flask import Flask
from flask_cors import CORS
from sockets import socketio
from api import blueprint as api
from multiprocessing import Process
from mast.interface.MastServer import MastServer
from api.components import send_queue, res_queue

import requests
import logging


def create_app():
    flask = Flask(__name__,
                  static_url_path='',
                  static_folder='dist')
    # mount all blueprints from api module.
    flask.register_blueprint(api)
    socketio.init_app(flask)
    cors = CORS(flask)
    return flask


if __name__ == '__main__':
    mast_server = MastServer()
    Process(target=mast_server.run, args=(send_queue, res_queue,)).start()
    app = create_app()
    socketio.run(app=app, host='0.0.0.0')

    # logger = logging.getLogger('gunicorn.error')
    # app.logger.handlers = logger.handlers
    # app.logger.setLevel(logger.level)

    # @app.route('/', defaults={'path': ''})
    # @app.route('/<path:path>')
    # def index(path):
    #     if app.debug:
    #         return requests.get('http://frontend:8080/{}'.format(path)).text
    #     return app.send_static_file('index.html')

    # flask app is recommended to be incorporated with gunicorn framework in production environment.
    # gunicorn -c webserver/gunicorn_config.py webserver:app --no-sendfile

    # but if you are testing in development environment, execute app.run() should be fine.
    # app.run()
