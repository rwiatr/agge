import os, psutil, time
import sys
sys.path.append(os.getcwd())

import threading
from experiment.ipinyou.onehot.data_manager import DataManager

def run_optuna(data, l, d, i):
    for x in range(4):
        time.sleep(1)
        print(f'LEN:    {len(data)}')
        print(f'thread {i}: process uses {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB of memory')

if __name__ == "__main__":
    # data
    subject = '2261'
    bins = 100
    sample_id = 1
    d_mgr = DataManager()
    # GET DATA
    data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, sample_id)
    print(f'After loading data the proces uses: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB of memory')

    #define threads
    thread_list = []
    for i in range(int(sys.argv[1])):
        try:
            thread_list += [threading.Thread(target=run_optuna, args=(data, linear_feature_columns, dnn_feature_columns, i))]
        except:
            print(f'unable to create thread {i}')

    #start defined threads
    for id, thread in enumerate(thread_list):
        try:
            thread.start()
        except:
            print('unable to start thread {id}')
    
    for thread_entity in thread_list:
        thread_entity.join()

    print('EVERYTHING HAS FINISHED')
    #run_optuna(data, linear_feature_columns, dnn_feature_columns)