from flask import Blueprint
from flask_restplus import Api

from .contents import api as ns_contents
from config import Config

# Create /api/ space
blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(
    blueprint,
    title=Config.NAME,
    version=Config.VERSION,
)

# mount related contents blueprint
api.add_namespace(ns_contents)

# mount related styles blueprint
# ...

# mount related stylizations blueprint
# ...
