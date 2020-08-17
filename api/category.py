#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: categories.py
@time: 2020/8/12 10:46
@version 1.0
@desc:
"""
import os

from flask_restplus import Namespace, Resource

from config import Config

api = Namespace('category', description='Category related operations')


# os.makedirs(Config.CATEGORIES_DIRECTORY, exist_ok=True)


@api.route('/')
class Category(Resource):

    def get(self):
        """ Returns all categories """
        # return query_util.fix_ids(current_user.categories.all())
        return [
            {
                'value': 'WebCaricature',
                'label': 'WebCaricature'
            },
            {
                'value': 'COCO',
                'label': 'COCO'
            },
        ]
