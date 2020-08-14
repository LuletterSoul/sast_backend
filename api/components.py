from multiprocessing import Manager

send_queue = Manager().Queue()
res_queue = Manager().Queue()
