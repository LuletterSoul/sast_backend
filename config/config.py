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

    # Dataset Options
    CONTENT_DIRECTORY = os.getenv("CONTENT_DIRECTORY", os.path.join(root_path, "data/contents"))
    STYLE_DIRECTORY = os.getenv("STYLE_DIRECTORY", os.path.join(root_path, "data/styles"))
    STYLIZATION_DIRECTORY = os.getenv("STYLIZATION_DIRECTORY", os.path.join(root_path, "data/stylizations"))
    LANDMARK_DIRECTORY = os.getenv("LANDMARK_DIRECTORY", os.path.join(root_path, "data/landmarks"))
    MAST_TOTAL_TIME = 1.5
    ANNOTATION_DIRECTORY = os.getenv("ANNOTATION_DIRECTORY", os.path.join(root_path, "data/annotations"))
    CATEGORIES_DIRECTORY = os.getenv("CATEGORIES_DIRECTORY", os.path.join(root_path, "data/categories"))

    # Redis
    REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
    REDIS_CELERY_BROKEN = os.getenv("REDIS_CELERY_BROKEN", "redis://localhost:6379/1")
    REDIS_RESULT_BACKEND = os.getenv("REDIS_RESULT_BACKEND", "redis://localhost:6379/2")
    REDIS_SOCKET_URL = os.getenv("REDIS_SOCKET_URL", "redis://localhost:6379/3")

    # MAST Options
    MAST_WORK_DIR = os.getenv("MAST_WORK_DIR", os.path.join(root_path, 'mast'))
    MAST_BATCH_SIZE = os.getenv("MAST_BATCH_SIZE", 1)
    MAST_WORKER_NUM = os.getenv("MAST_WORKER_NUM", 2)
    MAST_CHANNEL = os.getenv("MAST_CHANNEL", "mast")
    MAST_DEVICES = os.getenv("MAST_DEVICES", "1,3")

    # CAST Options
    # content_dir = os.getenv("content_dir", "images/content")
    # style_dir = os.getenv("style_dir", "images/style")
    # landmark_dir = os.getenv("landmark_dir", "images/landmark")
    # output_dir = os.getenv("output_dir", "output")
    # max_iter = 500
    # max_iter_hr = 200

    CAST_WORK_DIR = os.getenv("CAST_WORK_DIR", os.path.join(root_path, 'cast'))
    CAST_BATCH_SIZE = os.getenv("CAST_BATCH_SIZE", 1)
    CAST_WORKER_NUM = os.getenv("CAST_WORKER_NUM", 2)
    CAST_CHANNEL = os.getenv("CAST_CHANNEL", "cast")
    CAST_DEVICES = os.getenv("MAST_DEVICES", "1,3")

    ### DIST Options
    content_dir_dist = os.getenv("content_dir_dist", 'data/video/')
    style_dir_dist = os.getenv("style_dir_dist", "data/style/")
    output_dir_dist = os.getenv('output_dir_dist' , 'data/Video_Results')
    encoder_dir_dist = os.getenv('encoder_dir_dist' , 'dist/models/4SE.pth')
    decoder_dir_dist = os.getenv('decoder_dir_dist' , 'dist/models/4SD.pth')
    matrix_dir_dist = os.getenv('matrix_dir_dist' , 'dist/models/FTM.pth') 
    loadSize = 512
    fineSizeH = 512
    fineSizeW = 512
    gpu_id = 1

__all__ = ["Config"]
