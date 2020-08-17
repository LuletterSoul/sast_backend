import io
import os
import uuid

from PIL import Image
from flask import send_file
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from workers import mast_report

from config import Config

api = Namespace('stylizations', description='Stylizations related operations')
os.makedirs(Config.STYLIZATION_DIRECTORY, exist_ok=True)

image_all = reqparse.RequestParser()
image_all.add_argument('page', default=1, type=int)
image_all.add_argument('size', default=50, type=int, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument('file', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG file')

image_stylization = reqparse.RequestParser()
image_stylization.add_argument('content_id', type=str, required=True, location='json', help='Content image id.')
image_stylization.add_argument('style_id', type=str, required=True, location='json', help='Style image id.')
image_stylization.add_argument('alg', type=str, required=True, location='json', help='CAST | MAST')
image_stylization.add_argument('sid', type=str, required=True, location='json', help='socket session id.')
image_stylization.add_argument('width', type=int, required=True, location='json', help='Image width')
image_stylization.add_argument('height', type=int, required=True, location='json', help='Image height')
image_stylization.add_argument('content_mask', location='json',
                               type=list, required=False,
                               help='PNG or JPG file')
image_stylization.add_argument('style_mask', location='json',
                               type=list, required=False,
                               help='PNG or JPG file')

# image_stylization.add_argument('content_mask',type=list,location )
# create_annotation.add_argument('keypoints', type=list, location='json', default=[])
image_download = reqparse.RequestParser()
image_download.add_argument('asAttachment', type=bool, default=False)
image_download.add_argument('width', type=int, default=512)
image_download.add_argument('height', type=int, default=512)
image_download.add_argument('timestamp', type=str, default='')
image_download.add_argument('category', type=str, default=None)

from workers import RedisStreamer

mast_streamer = RedisStreamer(redis_broker=Config.REDIS_BROKER_URL, prefix='mast')
cast_streamer = RedisStreamer(redis_broker=Config.REDIS_BROKER_URL, prefix='cast')


@api.route('/')
class Stylizations(Resource):

    @api.expect(image_all)
    def get(self):
        """ Returns pageable content image"""
        args = image_all.parse_args()
        per_page = args['size']
        page = args['page'] - 1

        stylization_ids = os.listdir(Config.CONTENT_DIRECTORY)
        total = len(stylization_ids)
        pages = int(total / per_page)

        return {
            "total": total,
            "pages": pages,
            "page": page,
            "size": per_page,
            "stylization_ids": stylization_ids
        }

    @api.expect(image_stylization)
    def post(self):
        """ Creates an image """
        args = image_stylization.parse_args()
        content_id = args['content_id']
        style_id = args['style_id']
        width = args['width']
        height = args['height']
        alg = args['alg']
        sid = args['sid']

        content_mask = args['content_mask']
        style_mask = args['style_mask']
        # print(f'content_mask={content_mask}')
        # print(f'style_mask={style_mask}')

        # ...
        # execute MAST

        req_id = str(uuid.uuid1())
        msg = [{
            'sid': sid,
            'req_id': req_id,
            'content_id': content_id,
            'style_id': style_id,
            'width': width,
            'height': height,
            'content_mask': content_mask,
            'style_mask': style_mask
        }]
        if alg == 'MAST':
            mast_streamer.submit(msg)
        elif alg == 'CAST':
            cast_streamer.submit(msg)
        return


@api.route('/<stylization_id>')
class StylizationId(Resource):

    @api.expect(image_download)
    def get(self, stylization_id):
        """ Returns category by ID """
        args = image_download.parse_args()
        as_attachment = args.get('asAttachment')
        category = args.get('category')

        # get intermediate stylized image or not
        if category != 'original':
            path = os.path.join(Config.STYLIZATION_DIRECTORY, category, f'{stylization_id}')
        else:
            path = os.path.join(Config.STYLIZATION_DIRECTORY, f'{stylization_id}')

        pil_image = None
        if os.path.exists(path):
            pil_image = Image.open(path)
        if pil_image is None:
            return {'success': False}, 400

        # we need different size image by parameters passed from client end.
        width = args.get('width')
        height = args.get('height')

        if not width:
            width = pil_image.size[1]
        if not height:
            height = pil_image.size[0]

        img_filename = f'{stylization_id}'

        pil_image.thumbnail((width, height), Image.ANTIALIAS)
        image_io = io.BytesIO()
        pil_image.save(image_io, "PNG")
        image_io.seek(0)

        # complete all business logic codes here including image resizing and image transmission !

        # image must be resized by previous width and height
        # and I/O pipe must be built for bytes transmission between backend and client end
        return send_file(image_io, attachment_filename=img_filename, as_attachment=as_attachment)


