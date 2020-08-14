import os
import subprocess


# def get_tag():
#     result = subprocess.run(["git", "describe", "--abbrev=0", "--tags"], stdout=subprocess.PIPE)
#     return str(result.stdout.decode("utf-8")).strip()


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
    CONTENT_DIRECTORY = os.getenv("CONTENT_DIRECTORY", "data/contents")
    STYLE_DIRECTORY = os.getenv("STYLE_DIRECTORY", "data/styles")
    STYLIZATION_DIRECTORY = os.getenv("STYLIZATION_DIRECTORY", "data/stylizations")
    MAST_TOTAL_TIME = 5
    ANNOTATION_DIRECTORY = os.getenv("ANNOTATION_DIRECTORY", "data/annotations")
    CATEGORIES_DIRECTORY = os.getenv("CATEGORIES_DIRECTORY", "data/categories")


__all__ = ["Config"]
