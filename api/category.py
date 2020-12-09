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
                    'upload': True,
                    'disable': False
                },
                {
                    'value': 'CAST',
                    'label': 'CAST',
                    'upload': True,
                    'disable': False
                },
                {
                    'value': 'DIST',
                    'label': 'DIST',
                    'upload': True,
                    'disable': False
                }
            ],
            'category_options':
                [
                    {
                        'value': 'WebCaricature',
                        'label': '漫画',
                        'disable': False
                    },
                    {
                        'value': 'COCO',
                        'label': '生活',
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': '油画',
                        'disable': False
                    },
                    {
                        'value': 'Video',
                        'label': '色彩',
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
                    },
                    {
                        'value': 'DIST',
                        'label': 'DIST',
                        'disable': True
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
                    },
                    {
                        'value': 'DIST',
                        'label': 'DIST',
                        'disable': True
                    }
                ],
                'Video': [
                    {
                        'value': 'MAST',
                        'label': 'MAST',
                        'disable': True
                    },
                    {
                        'value': 'CAST',
                        'label': 'CAST',
                        'disable': True
                    },
                    {
                        'value': 'DIST',
                        'label': 'DIST',
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
                    },
                    {
                        'value': 'DIST',
                        'label': 'DIST',
                        'disable': True
                    }
                ],
            },
            'alg_compatible_map': {
                'CAST': [
                    {
                        'value': 'COCO',
                        'label': '生活',
                        'upload': False,
                        'disable': True
                    },
                    {
                        'value': 'WebCaricature',
                        'label': '漫画',
                        'upload': False,
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': '油画',
                        'upload': False,
                        'disable': False
                    },
                    {
                        'value': 'Video',
                        'label': '色彩',
                        'upload': False,
                        'disable': True
                    },
                ],
                'MAST': [
                    {
                        'value': 'WebCaricature',
                        'label': '漫画',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'COCO',
                        'label': '生活',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': '油画',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'Video',
                        'label': '色彩',
                        'upload': True,
                        'disable': True
                    },
                ],
                'DIST': [
                    {
                        'value': 'WebCaricature',
                        'label': '漫画',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'COCO',
                        'label': '生活',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'ArtisticFaces',
                        'label': '油画',
                        'upload': True,
                        'disable': False
                    },
                    {
                        'value': 'Video',
                        'label': '色彩',
                        'upload': True,
                        'disable': False
                    },
                ]
            }
        }
