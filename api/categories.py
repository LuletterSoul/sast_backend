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
import json
import os

from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user
from mongoengine.errors import NotUniqueError

import datetime

from config import Config

api = Namespace('category', description='Category related operations')
os.makedirs(Config.CATEGORIES_DIRECTORY, exist_ok=True)

create_category = reqparse.RequestParser()
create_category.add_argument('name', required=True, location='json')
create_category.add_argument('supercategory', location='json')
create_category.add_argument('color', location='json')
create_category.add_argument('metadata', type=dict, location='json')
create_category.add_argument(
    'keypoint_edges', type=list, default=[], location='json')
create_category.add_argument(
    'keypoint_labels', type=list, default=[], location='json')
create_category.add_argument(
    'keypoint_colors', type=list, default=[], location='json')

update_category = reqparse.RequestParser()
update_category.add_argument('name', required=True, location='json')
update_category.add_argument('supercategory', location='json')
update_category.add_argument('color', location='json')
update_category.add_argument('metadata', type=dict, location='json')
update_category.add_argument('keypoint_edges', type=list, location='json')
update_category.add_argument('keypoint_labels', type=list, location='json')
update_category.add_argument('keypoint_colors', type=list, location='json')

page_data = reqparse.RequestParser()
page_data.add_argument('page', default=1, type=int)
page_data.add_argument('limit', default=20, type=int)


@api.route('/')
class Category(Resource):

    def get(self):
        """ Returns all categories """
        # return query_util.fix_ids(current_user.categories.all())
        category_ids = os.listdir(Config.CATEGORIES_DIRECTORY)
        categories = []
        for c in category_ids:
            categories.append(json.load(open(os.path.join(Config.CATEGORIES_DIRECTORY, c))))
        return categories

    @api.expect(create_category)
    def post(self):
        """ Creates a category """
        args = create_category.parse_args()
        name = args.get('name')
        supercategory = args.get('supercategory')
        metadata = args.get('metadata', {})
        color = args.get('color')
        keypoint_edges = args.get('keypoint_edges')
        keypoint_labels = args.get('keypoint_labels')
        keypoint_colors = args.get('keypoint_colors')

        category_id = len(os.listdir(Config.CATEGORIES_DIRECTORY))

        try:
            category = {
                'name': name,
                'supercategory': supercategory,
                'color': color,
                'metadata': metadata,
                'keypoint_edges': keypoint_edges,
                'keypoint_labels': keypoint_labels,
                'keypoint_colors': keypoint_colors,
            }
            with open(os.path.join(Config.CATEGORIES_DIRECTORY, f'{category_id}.json'), 'w') as f:
                json.dump(category, f)
        except NotUniqueError as e:
            return {'message': 'Category already exists. Check the undo tab to fully delete the category.'}, 400
        return category


@api.route('/<int:category_id>')
class Category(Resource):

    @login_required
    def get(self, category_id):
        """ Returns a category by ID """
        # category = current_user.categories.filter(id=category_id).first()
        category = json.load(open(os.path.join(Config.CATEGORIES_DIRECTORY, category_id)))

        if category is None:
            return {'success': False}, 400

        # return query_util.fix_ids(category)
        return category

    @login_required
    def delete(self, category_id):
        """ Deletes a category by ID """
        category = current_user.categories.filter(id=category_id).first()
        if category is None:
            return {"message": "Invalid image id"}, 400

        if not current_user.can_delete(category):
            return {"message": "You do not have permission to delete this category"}, 403

        category.update(set__deleted=True,
                        set__deleted_date=datetime.datetime.now())
        return {'success': True}

    @api.expect(update_category)
    @login_required
    def put(self, category_id):
        """ Updates a category name by ID """

        category = current_user.categories.filter(id=category_id).first()

        # check if the id exits
        if category is None:
            return {"message": "Invalid category id"}, 400

        args = update_category.parse_args()
        name = args.get('name')
        supercategory = args.get('supercategory', category.supercategory)
        color = args.get('color', category.color)
        metadata = args.get('metadata', category.metadata)
        keypoint_edges = args.get('keypoint_edges', category.keypoint_edges)
        keypoint_labels = args.get('keypoint_labels', category.keypoint_labels)
        keypoint_colors = args.get('keypoint_colors', category.keypoint_colors)

        # check if there is anything to update
        if category.name == name \
                and category.supercategory == supercategory \
                and category.color == color \
                and category.keypoint_edges == keypoint_edges \
                and category.keypoint_labels == keypoint_labels \
                and category.keypoint_colors == keypoint_colors:
            return {"message": "Nothing to update"}, 200

        # check if the name is empty
        if not name:
            return {"message": "Invalid category name to update"}, 400

        # update name of the category
        # check if the name to update exits already in db
        # @ToDo: Is it necessary to allow equal category names among different creators?
        category.name = name
        category.supercategory = supercategory
        category.color = color
        category.keypoint_edges = keypoint_edges
        category.keypoint_labels = keypoint_labels
        category.keypoint_colors = keypoint_colors

        try:
            category.update(
                name=category.name,
                supercategory=category.supercategory,
                color=category.color,
                metadata=category.metadata,
                keypoint_edges=category.keypoint_edges,
                keypoint_labels=category.keypoint_labels,
                keypoint_colors=category.keypoint_colors,
            )
        except NotUniqueError:
            # it is only triggered when the name already exists and the creator is the same
            return {"message": "Category '" + name_to_update + "' already exits"}, 400

        return {"success": True}


@api.route('/data')
class CategoriesData(Resource):

    @api.expect(page_data)
    @login_required
    def get(self):
        """ Endpoint called by category viewer client """
        pass
        # args = page_data.parse_args()
        # limit = args['limit']
        # page = args['page']
        #
        # categories = current_user.categories.filter(deleted=False)
        #
        # pagination = Pagination(categories.count(), limit, page)
        # categories = query_util.fix_ids(
        #     categories[pagination.start:pagination.end])
        #
        # for category in categories:
        #     category['numberAnnotations'] = AnnotationModel.objects(
        #         deleted=False, category_id=category.get('id')).count()
        #
        # return {
        #     "pagination": pagination.export(),
        #     "page": page,
        #     "categories": categories
        # }
