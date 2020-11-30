# sast_backend
 The backend services of semantically aligned style transfer system.
 
# Requirements


# Redis

This project requires message broker based Redis.

    git clone https://github.com/redis/redis.git
    cd redis
    sudo make
    sudo make install
    cd src
    ./redis-server

Default redis listening on port 6379.


# Celery 

We construct task queues that are used as a mechanism to distribute style transfer requests across threads or machines. Celery handle all asynchronous tasks while executing time-consuming style transfer procedure. Run celery in the command line as below:

    celery -A workers worker -l info


## python version
python:  3.6+
## install dependencies


``
pip install -r requirements.txt
``

# Run backend in test environment

``
python app.py
``

