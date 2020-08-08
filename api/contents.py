from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user
from werkzeug.datastructures import FileStorage
from flask import send_file

from config import Config
from PIL import Image
import datetime
import os
import io

api = Namespace('contents', description='Contents related operations')

image_all = reqparse.RequestParser()
image_all.add_argument('page', default=1, type=int)
image_all.add_argument('size', default=50, type=int, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument('image', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG file')

image_download = reqparse.RequestParser()
image_download.add_argument('asAttachment', type=bool, default=False)
image_download.add_argument('width', type=int)
image_download.add_argument('height', type=int)


@api.route('/')
class Contents(Resource):

    @api.expect(image_all)
    def gets(self):
        """ Returns pageable content image"""
        args = image_all.parse_args()
        per_page = args['size']
        page = args['page'] - 1

        # return {
        #     "total": total,
        #     "pages": pages,
        #     "page": page,
        #     "size": per_page,
        #     "content_ids": [content_id1,content_id2]
        # }

    @api.expect(image_upload)
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args['image']

        directory = Config.CONTENT_DIRECTORY
        path = os.path.join(directory, image.filename)

        if os.path.exists(path):
            return {'message': 'file already exists'}, 400

        pil_image = Image.open(io.BytesIO(image.read()))

        pil_image.save(path)

        image.close()
        pil_image.close()
        return


@api.route('/<int:content_id>')
class ContentId(Resource):

    @api.expect(image_download)
    def get(self, content_id):
        """ Returns category by ID """
        args = image_download.parse_args()
        as_attachment = args.get('asAttachment')

        # Here content image should be loaded from corresponding directory.
        image = None

        if image is None:
            return {'success': False}, 400

        # we need different size image by parameters passed from client end.
        width = args.get('width')
        height = args.get('height')

        if not width:
            width = image.width
        if not height:
            height = image.height

        # complete all business logic codes here including image resizing and image transmission !

        # image must be resized by previous width and height
        # and I/O pipe must be built for bytes transmission between backend and client end
        # return send_file(image_io, attachment_filename=image.file_name, as_attachment=as_attachment)
