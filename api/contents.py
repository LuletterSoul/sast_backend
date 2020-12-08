from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user
from werkzeug.datastructures import FileStorage
from flask import send_file
from utils import *

from config import Config
from PIL import Image
import datetime
import os
import io

api = Namespace('contents', description='Contents related operations')
# create data storage directory
os.makedirs(Config.CONTENT_DIRECTORY, exist_ok=True)
# os.makedirs(Config.CAST_DATA_DIR, exist_ok=True)

image_all = reqparse.RequestParser()
image_all.add_argument('page', default=0, type=int)
image_all.add_argument('size', default=50, type=int, required=False)
image_all.add_argument('category', default='', type=str, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument('category', location='args',
                          type=str, default='COCO',
                          help='File category')
image_upload.add_argument('file', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG file')

image_download = reqparse.RequestParser()
image_download.add_argument('asAttachment', type=bool, default=False)
image_download.add_argument('width', type=int, default=512)
image_download.add_argument('height', type=int, default=512)
image_download.add_argument('category', default='', type=str, required=False)
image_download.add_argument('style_category', default='', type=str, required=False)
image_download.add_argument('content_category', default='', type=str, required=False)
image_download.add_argument(
    'videoType', default='preview', type=str, required=False)


@api.route('/')
class Contents(Resource):

    @api.expect(image_all)
    def get(self):
        """ Returns pageable content image"""
        args = image_all.parse_args()
        size = args['size']
        page = args['page']
        category = args['category']

        path = os.path.join(Config.CONTENT_DIRECTORY, category)

        if not os.path.exists(path):
            content_ids = []
        else:
            content_ids = [p for p in os.listdir(path) if os.path.isfile(os.path.join(path,p))]
        
        total = len(content_ids)
        pages = int(total / size)

        page_content_ids = []

        if (page + 1) * size > total and page * size < total:
            page_content_ids = content_ids[page*size:]
        else:
            page_content_ids = content_ids[page * size:(page+1)*size]


        return {
            "total": total,
            "pages": pages,
            "page": page,
            "size": size,
            "content_ids": page_content_ids 
        }

    @api.expect(image_upload)
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args['file']
        category = args['category']

        print(category)

        directory = os.path.join(Config.CONTENT_DIRECTORY, category)
        os.makedirs(directory,exist_ok=True)
        path = os.path.join(directory,image.filename)

        # if os.path.exists(path):
        #     return {'message': 'file already exists'}, 400

        pil_image = Image.open(io.BytesIO(image.read()))

        pil_image.save(path)

        image.close()
        pil_image.close()
        return image.filename


@api.route('/<content_id>')
class ContentId(Resource):

    @api.expect(image_download)
    def get(self, content_id):
        """ Returns category by ID """
        args = image_download.parse_args()
        as_attachment = args.get('asAttachment')
        category = args.get('category')

        content_name = os.path.splitext(content_id)[0]
        fmt = os.path.splitext(content_id)[1].lower()

        path = os.path.join(Config.CONTENT_DIRECTORY,
                            category, f'{content_id}')
        if not os.path.exists(path):
            print(f'Content do not exist in {path}')
            return

        width = args.get('width')
        height = args.get('height')

        if is_photo(fmt):
            return send_img(path, content_id, width, height, as_attachment)

        elif is_video(fmt):
            video_type = args.get('videoType')
            if video_type == 'preview':
                preview_path = os.path.join(Config.CONTENT_DIRECTORY,
                                            category, content_name, 'preview.png')
                return send_img(preview_path, 'preview.png', width, height, as_attachment)
            elif video_type == 'video':
                return send_file(path, attachment_filename=content_id, as_attachment=as_attachment)
