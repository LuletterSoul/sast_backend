from celery import Celery
from config import Config
from .stream import *
from .tasks import *

# connect_mongo('Celery_Worker')

celery = Celery(
    Config.NAME,
    broker=Config.REDIS_CELERY_BROKEN,
)

# streamer = RedisStreamer()

if __name__ == '__main__':
    celery.start()
