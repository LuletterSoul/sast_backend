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
        return {
            'alg_default': 'MAST',
            'category_default': 'COCO',
            'alg_options': [
                {
                    'value': 'MAST',
                    'label': 'MAST',
                    'disable': False
                },
                {
                    'value': 'CAST',
                    'label': 'CAST',
                    'disable': False
                }
            ],
            'category_options':
                [
                    {
                        'value': 'WebCaricature',
                        'label': 'WebCaricature',
                        'disable': False
                    },
                    {
                        'value': 'COCO',
                        'label': 'COCO',
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': 'ArtisticFaces',
                        'disable': False
                    },
                ],
            'dataset_compatible_map': {
                'WebCaricature': [
                    {
                        'value': 'MAST',
                        'label': 'MAST',
                        'disable': False
                    },
                    {
                        'value': 'CAST',
                        'label': 'CAST',
                        'disable': False
                    }
                ],
                'ArtisticFaces': [
                    {
                        'value': 'MAST',
                        'label': 'MAST',
                        'disable': False
                    },
                    {
                        'value': 'CAST',
                        'label': 'CAST',
                        'disable': False
                    }
                ],
                'COCO': [
                    {
                        'value': 'MAST',
                        'label': 'MAST',
                        'disable': False
                    },
                    {
                        'value': 'CAST',
                        'label': 'CAST',
                        'disable': True
                    }
                ],
            },
            'alg_compatible_map': {
                'CAST': [
                    {
                        'value': 'COCO',
                        'label': 'COCO',
                        'disable': True
                    },
                    {
                        'value': 'WebCaricature',
                        'label': 'WebCaricature',
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': 'ArtisticFaces',
                        'disable': False
                    },
                ],
                'MAST': [
                    {
                        'value': 'WebCaricature',
                        'label': 'WebCaricature',
                        'disable': False
                    },
                    {
                        'value': 'COCO',
                        'label': 'COCO',
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': 'ArtisticFaces',
                        'disable': False
                    },
                ]
            }
        }
