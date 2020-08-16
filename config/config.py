import os
import subprocess

# def get_tag():
#     result = subprocess.run(["git", "describe", "--abbrev=0", "--tags"], stdout=subprocess.PIPE)
#     return str(result.stdout.decode("utf-8")).strip()
project_name = 'sast_backend'
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path[:cur_path.find(project_name + os.sep) + len(project_name + os.sep)]


def _get_bool(key, default_value):
    if key in os.environ:
        value = os.environ[key]
        if value == 'True' or value == 'true' or value == '1':
            return True
        return False
    return default_value


class Config:
    NAME = os.getenv("NAME", "SAST System")
    # VERSION = get_tag()
    VERSION = 'v1.0'

    ### Dataset Options
    CONTENT_DIRECTORY = os.getenv("CONTENT_DIRECTORY", os.path.join(root_path, "data/contents"))
    STYLE_DIRECTORY = os.getenv("STYLE_DIRECTORY", os.path.join(root_path, "data/styles"))
    STYLIZATION_DIRECTORY = os.getenv("STYLIZATION_DIRECTORY", os.path.join(root_path, "data/stylizations"))
    MAST_TOTAL_TIME = 1.5
    ANNOTATION_DIRECTORY = os.getenv("ANNOTATION_DIRECTORY", os.path.join(root_path, "data/annotations"))
    CATEGORIES_DIRECTORY = os.getenv("CATEGORIES_DIRECTORY", os.path.join(root_path, "data/categories"))

    ## Redis
    REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
    REDIS_RESULT_BACKEND = os.getenv("REDIS_RESULT_BACKEND", "mongodb://database/flask")


__all__ = ["Config"]
