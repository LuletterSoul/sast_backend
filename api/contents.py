from flask_restx import Namespace, Resource, reqparse
from flask_login import login_required, current_user
from werkzeug.datastructures import FileStorage
from flask import send_file

from config import Config
from PIL import Image
import datetime
import os
import io

api = Namespace('contents', description='Contents related operations')
# create data storage directory
os.makedirs(Config.CONTENT_DIRECTORY, exist_ok=True)

image_all = reqparse.RequestParser()
image_all.add_argument('page', default=1, type=int)
image_all.add_argument('size', default=50, type=int, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument('file', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG file')

image_download = reqparse.RequestParser()
image_download.add_argument('asAttachment', type=bool, default=False)
image_download.add_argument('width', type=int, default=512)
image_download.add_argument('height', type=int, default=512)


@api.route('/')
class Contents(Resource):

    @api.expect(image_all)
    def get(self):
        """ Returns pageable content image"""
        args = image_all.parse_args()
        per_page = args['size']
        page = args['page'] - 1

        content_ids = os.listdir(Config.CONTENT_DIRECTORY)
        total = len(content_ids)
        pages = int(total / per_page)

        return {
            "total": total,
            "pages": pages,
            "page": page,
            "size": per_page,
            "content_ids": content_ids
        }

    @api.expect(image_upload)
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args['file']

        directory = Config.CONTENT_DIRECTORY
        path = os.path.join(directory, image.filename)

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

        # Here content image should be loaded from corresponding directory.
        # image = None
        #
        pil_image = Image.open(os.path.join(Config.CONTENT_DIRECTORY, f'{content_id}'))

        if pil_image is None:
            return {'success': False}, 400

        # we need different size image by parameters passed from client end.
        width = args.get('width')
        height = args.get('height')

        if not width:
            width = pil_image.size[1]
        if not height:
            height = pil_image.size[0]

        img_filename = f'{content_id}.png'

        pil_image.thumbnail((width, height), Image.ANTIALIAS)
        image_io = io.BytesIO()
        pil_image.save(image_io, "PNG")
        image_io.seek(0)

        # complete all business logic codes here including image resizing and image transmission !

        # image must be resized by previous width and height
        # and I/O pipe must be built for bytes transmission between backend and client end
        return send_file(image_io, attachment_filename=img_filename, as_attachment=as_attachment)
