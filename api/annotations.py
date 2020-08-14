#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: annotations.py
@time: 2020/8/12 10:20
@version 1.0
@desc:
"""
import os

from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user

import datetime
import logging
import json

from config import Config

logger = logging.getLogger('gunicorn.error')

api = Namespace('annotations', description='Annotation related operations')
os.makedirs(Config.ANNOTATION_DIRECTORY, exist_ok=True)

create_annotation = reqparse.RequestParser()
create_annotation.add_argument(
    'image_id', type=str, required=True, location='json')
create_annotation.add_argument('category_id', type=str, location='json')
create_annotation.add_argument('isbbox', type=bool, location='json', default=False)
create_annotation.add_argument('metadata', type=dict, location='json', default={})
create_annotation.add_argument('segmentation', type=list, location='json', default=[])
create_annotation.add_argument('keypoints', type=list, location='json', default=[])
create_annotation.add_argument('color', location='json')

update_annotation = reqparse.RequestParser()
update_annotation.add_argument('category_id', type=int, location='json')


@api.route('/')
class Annotation(Resource):

    # @login_required
    def get(self):
        """ Returns all annotations """
        # return query_util.fix_ids(current_user.annotations.exclude("paper_object").all())
        pass

    @api.expect(create_annotation)
    def post(self):
        """ Creates an annotation """
        args = create_annotation.parse_args()
        image_id = args.get('image_id')
        category_id = args.get('category_id')
        isbbox = args.get('isbbox')
        metadata = args.get('metadata', {})
        segmentation = args.get('segmentation', [])
        keypoints = args.get('keypoints', [])

        # image = current_user.images.filter(id=image_id, deleted=False).first()
        # if image is None:
        #     return {"message": "Invalid image id"}, 400
        #
        # logger.info(
        #     f'{current_user.username} has created an annotation for image {image_id} with {isbbox}')
        # logger.info(
        #     f'{current_user.username} has created an annotation for image {image_id}')

        # try:
        #     annotation = AnnotationModel(
        #         image_id=image_id,
        #         category_id=category_id,
        #         metadata=metadata,
        #         segmentation=segmentation,
        #         keypoints=keypoints,
        #         isbbox=isbbox
        #     )
        #     annotation.save()
        annotation = {
            'id': f'{os.path.splitext(image_id)[0]}_{os.path.splitext(category_id)[0]}',
            'image_id': image_id,
            'category_id': category_id,
            'metadata': metadata,
            'segmentation': segmentation,
            'keypoints': keypoints,
            'isbbox': isbbox
        }
        with open(f'{Config.ANNOTATION_DIRECTORY}/{os.path.splitext(image_id)[0]}.json', 'w') as f:
            json.dump(annotation, f)
        return annotation


@api.route('/<annotation_id>')
class AnnotationId(Resource):

    def get(self, annotation_id):
        """ Returns annotation by ID """
        # annotation = current_user.annotations.filter(id=annotation_id).first()

        # if annotation is None:
        #     return {"message": "Invalid annotation id"}, 400

        # return query_util.fix_ids(annotation)

    def delete(self, annotation_id):
        """ Deletes an annotation by ID """
        annotation = current_user.annotations.filter(id=annotation_id).first()

        if annotation is None:
            return {"message": "Invalid annotation id"}, 400

        image = current_user.images.filter(
            id=annotation.image_id, deleted=False).first()
        image.flag_thumbnail()

        annotation.update(set__deleted=True,
                          set__deleted_date=datetime.datetime.now())
        return {'success': True}

    # @api.expect(update_annotation)
    # @login_required
    # def put(self, annotation_id):
    #     """ Updates an annotation by ID """
    #     annotation = current_user.annotations.filter(id=annotation_id).first()
    #
    #     if annotation is None:
    #         return {"message": "Invalid annotation id"}, 400
    #
    #     args = update_annotation.parse_args()
    #
    #     new_category_id = args.get('category_id')
    #     annotation.update(category_id=new_category_id)
    #     logger.info(
    #         f'{current_user.username} has updated category for annotation (id: {annotation.id})'
    #     )
    #     newAnnotation = current_user.annotations.filter(id=annotation_id).first()
    #     return query_util.fix_ids(newAnnotation)

# @api.route('/<int:annotation_id>/mask')
# class AnnotationMask(Resource):
#     def get(self, annotation_id):
#         """ Returns the binary mask of an annotation """
#         return query_util.fix_ids(AnnotationModel.objects(id=annotation_id).first())
