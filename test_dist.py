from workers.tasks.dist import *

if __name__ == "__main__":
    msg = {
        'video_dir_id': 'data/video/temple_1/',
        'style_id': 'data/style/bamboo_forest.jpg'
    }
    dist_model = DISTModel()
    dist_model.init_models()
    dist_model.predict(msg)


