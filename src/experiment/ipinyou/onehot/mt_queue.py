from email.utils import quote
import queue
import threading
import pandas as pd
import sys
import time
import random
import datetime

thread_exit_flag = 0
options = {'loops': 4}


def some_funky_func(thread_id, results_list, data_int, q):
    while not thread_exit_flag:
        queue_lock.acquire()
        if not work_queue.empty():
            experiment = q.get()
            queue_lock.release()
            
            experiment_start_timestamp = time.time()
            total_auc = 0.0
            count = 0
            for d in range(data_int):
                for i in range(options['loops']):
                    time.sleep(1)
                sleep_time = random.randint(1, 3)
                print(f'{d}EXP |{experiment.upper()}| thread {thread_id} goes to sleep for {sleep_time} s')
                time.sleep(sleep_time)
                total_auc += random.random()
                count += 1
            
            experiment_finish_timestamp = time.time()
            auc = total_auc/count
            new_row = {"thread_id": thread_id, "Start": datetime.datetime.fromtimestamp(experiment_start_timestamp).strftime('%c'), "Finish": datetime.datetime.fromtimestamp(experiment_finish_timestamp).strftime('%c'), "Auc": auc}
            results_list.append(new_row)

        else:
            queue_lock.release()
            time.sleep(1)


if __name__ == "__main__":


    if sys.argv[1] == None:
        thread_amount = 1
    else:
        thread_amount = int(sys.argv[1])

    data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm','n','o', 'p', 'q', 'r' ,'s', 't', 'u', 'v', 'w', 's', 'y', 'z']
    results_list = []
    queue_lock = threading.Lock()
    work_queue = queue.Queue()

    #define threads
    data_int = 4
    thread_list = []
    for i in range(thread_amount):
        try:
            thread_list += [threading.Thread(target=some_funky_func, args=(i, results_list, data_int, work_queue))]
        except:
            print(f'unable to create thread {i}')

    #start defined threads
    for id, thread in enumerate(thread_list):
        try:
            thread.start()
        except:
            print(f'unable to start thread {id}')

    queue_lock.acquire()

    for item in data:
        work_queue.put(item)

    queue_lock.release()

    while not work_queue.empty():
        time.sleep(1)

    thread_exit_flag = 1

    for thread_entity in thread_list:
        thread_entity.join()
