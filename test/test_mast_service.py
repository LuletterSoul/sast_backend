import time
from multiprocessing import Queue, Process
import sys
sys.path.append('../')
from mast.interface.MastService import MastService


def receive_msg(results_queue: Queue):
    while True:
        if not results_queue.empty():
            msg = results_queue.get()
            content_img_id = msg['content_img_id']
            style_img_id = msg['style_img_id']
            stylized_img_id = msg['stylized_img_id']
            process_step = msg['process_step']
            print(f'[Receive Process]: have received the results from mast service, '
                  f'content_img_id={content_img_id} | style_img_id={style_img_id} | '
                  f'stylized_img_id={stylized_img_id} | process_step={process_step}')


def send_msg(receive_queue: Queue):
    content_img_id = '0.bmp'
    style_img_id = '0.bmp'
    while True:
        msg = {
            'content_img_id': content_img_id,
            'style_img_id': style_img_id,
        }
        receive_queue.put(msg)
        print(f'[Send Process]: have put the msg into receive queue, '
              f'content_img_id={content_img_id} | style_img_id={style_img_id}')
        time.sleep(10)


def test_mast_service():
    mast_service = MastService()
    receive_queue = Queue()
    results_queue = Queue()
    Process(target=mast_service.run, args=(receive_queue, results_queue,)).start()
    Process(target=send_msg, args=(receive_queue,)).start()
    Process(target=receive_msg, args=(results_queue,)).start()


if __name__ == '__main__':
    test_mast_service()
