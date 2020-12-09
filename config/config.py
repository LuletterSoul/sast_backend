import os
import subprocess

# def get_tag():
#     result = subprocess.run(["git", "describe", "--abbrev=0", "--tags"], stdout=subprocess.PIPE)
#     return str(result.stdout.decode("utf-8")).strip()
project_name = 'sast_backend'
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path[:cur_path.find(
    project_name + os.sep) + len(project_name + os.sep)]


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
    CONTENT_DIRECTORY = os.getenv(
        "CONTENT_DIRECTORY", os.path.join(root_path, "data/contents"))
    STYLE_DIRECTORY = os.getenv(
        "STYLE_DIRECTORY", os.path.join(root_path, "data/styles"))
    STYLIZATION_DIRECTORY = os.getenv(
        "STYLIZATION_DIRECTORY", os.path.join(root_path, "data/stylizations"))
    WARP_DIRECTORY = os.getenv(
        "WARP_DIRECTORY", os.path.join(root_path, "data/warp"))
    VIDEO_DIRECTORY = os.getenv(
        "VIDEO_DIRECTORY", os.path.join(root_path, "data/videos"))
    LANDMARK_DIRECTORY = os.getenv(
        "LANDMARK_DIRECTORY", os.path.join(root_path, "data/landmarks"))
    MAST_TOTAL_TIME = 1.5
    ANNOTATION_DIRECTORY = os.getenv(
        "ANNOTATION_DIRECTORY", os.path.join(root_path, "data/annotations"))
    CATEGORIES_DIRECTORY = os.getenv(
        "CATEGORIES_DIRECTORY", os.path.join(root_path, "data/categories"))

    # Redis
    REDIS_BROKER_URL = os.getenv(
        "REDIS_BROKER_URL", "redis://localhost:6379/0")
    REDIS_CELERY_BROKEN = os.getenv(
        "REDIS_CELERY_BROKEN", "redis://localhost:6379/1")
    REDIS_RESULT_BACKEND = os.getenv(
        "REDIS_RESULT_BACKEND", "redis://localhost:6379/2")
    REDIS_SOCKET_URL = os.getenv(
        "REDIS_SOCKET_URL", "redis://localhost:6379/3")

    # MAST Options
    MAST_WORK_DIR = os.getenv("MAST_WORK_DIR", os.path.join(root_path, 'mast'))
    MAST_BATCH_SIZE = os.getenv("MAST_BATCH_SIZE", 1)
    MAST_WORKER_NUM = os.getenv("MAST_WORKER_NUM", 1)
    MAST_CHANNEL = os.getenv("MAST_CHANNEL", "mast")
    MAST_DEVICES = os.getenv("MAST_DEVICES", "0,1")

    # CAST Options
    # content_dir = os.getenv("content_dir", "images/content")
    # style_dir = os.getenv("style_dir", "images/style")
    # landmark_dir = os.getenv("landmark_dir", "images/landmark")
    # output_dir = os.getenv("output_dir", "output")
    # max_iter = 500
    # max_iter_hr = 200

    CAST_WORK_DIR = os.getenv("CAST_WORK_DIR", os.path.join(root_path, 'cast'))
    CAST_BATCH_SIZE = os.getenv("CAST_BATCH_SIZE", 1)
    CAST_WORKER_NUM = os.getenv("CAST_WORKER_NUM", 1)
    CAST_CHANNEL = os.getenv("CAST_CHANNEL", "cast")
    CAST_DEVICES = os.getenv("MAST_DEVICES", "0,1")

    # DIST Options
    DIST_WORK_DIR = os.getenv("DIST_WORK_DIR", os.path.join(root_path, 'dist'))
    DIST_BATCH_SIZE = os.getenv("DIST_BATCH_SIZE", 1)
    DIST_WORKER_NUM = os.getenv("DIST_WORKER_NUM", 1)
    DIST_CHANNEL = os.getenv("DIST_CHANNEL", "dist")
    DIST_ENCODER = os.getenv('DIST_ENCODER', os.path.join(
        DIST_WORK_DIR, 'models/4SE.pth'))
    DIST_DECODER = os.getenv('DIST_DECODER', os.path.join(
        DIST_WORK_DIR, 'models/4SD.pth'))
    DIST_MATRIX = os.getenv('DIST_MATRIX', os.path.join(
        DIST_WORK_DIR, 'models/FTM.pth'))
    LOAD_SIZE = 512
    FINE_SIZE_H = 512
    FINE_SIZE_W = 1024
    DIST_DEVICES = os.getenv('DIST_DEVICES', "0,1")


__all__ = ["Config"]
