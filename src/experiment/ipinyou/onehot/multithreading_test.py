import os, psutil, time
import sys
sys.path.append(os.getcwd())

import threading
from experiment.ipinyou.onehot.data_manager import DataManager
from experiment.ipinyou.onehot.run import generate_space
from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
from math import ceil
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import copy
import queue

OPTIONS = {
    'n_datasets': 2,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
thread_exit_flag = 0

def create_directiories(dirs = [f'./mt_data/threads_{sys.argv[1]}', './models_mt']):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f'directory {dir} has been created')
        else:
            print(f'directory {dir} already exists')

def train_evalute_model(subject, X, y , linear_feature_columns, dnn_feature_columns, id, **properties):
    
    es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=0, patience=properties["n_iter_no_change"], mode='min')
    mdckpt = ModelCheckpoint(filepath=f'./models_mt/model_id{id}.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if properties['define_new_model']:
        model = DCN(
            linear_feature_columns = linear_feature_columns,
            dnn_feature_columns= dnn_feature_columns, 
            dnn_hidden_units=properties['hidden_layer_sizes'], 
            task='binary',
            l2_reg_embedding=properties['l2_reg'], 
            device=device, 
            dnn_dropout=properties['dnn_dropout'])
    else:
        model = torch.load(f'./models_mt/model_id{id}.ckpt')

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=properties['learning_rate_init'],
        betas=(properties['beta_1'], properties['beta_2']),
        eps=properties['epsilon'],
        weight_decay=properties['alpha'])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])
    
    current_lr_value = properties['learning_rate_init']
    #print(f'Model evaluation for lr={current_lr_value}')

    history = model.fit(
        x=X['train'], 
        y=y['train'], 
        batch_size=properties['batch_size'], 
        epochs=properties['max_iter'], 
        verbose=0,
        validation_data=(X['vali'], y['vali']),
        shuffle=False,
        callbacks=[es, mdckpt])

    loss_arr = history.history['loss']
    #print(f'EPOCHS: {len(loss_arr)}')
    model_best = torch.load(f'./models_mt/model_id{id}.ckpt')

    train_auc = round(roc_auc_score(y['train'], model.predict(X['train'], properties['batch_size'])), 4)
    test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], properties['batch_size'])), 4)
    best_train_auc = round(roc_auc_score(y['train'], model_best.predict(X['train'], properties['batch_size'])), 4)
    best_test_auc = round(roc_auc_score(y['test'], model_best.predict(X['test'], properties['batch_size'])), 4)
    #print(f"TRAIN_AUC: {train_auc}, TEST_AUC: {test_auc}, BEST_TEST_AUC: {best_test_auc} ")
    results = {"TRAIN_AUC": train_auc, "TEST_AUC": test_auc, "BEST_TEST_AUC": best_test_auc, "BEST_TRAIN_AUC": best_train_auc}
    return results

    

def run_learning_thread(data_list, thread_id, nn_params, global_data_list, q):

    
    while not thread_exit_flag:
        queue_lock.acquire()
        if not work_queue.empty():
            
            experiment_id, (subject, sample_id, alpha, lr, batch_size, hidden, dnn_dropout, l2_reg) = q.get()
            queue_lock.release()

            start = time.time()
            nn_params = copy.deepcopy(nn_params)
            data_list = copy.deepcopy(data_list)
            count = 0

            hyperparameters = {
                "hidden_layer_sizes": hidden,
                "alpha": alpha,
                "learning_rate_init": lr,
                "batch_size": batch_size,
                "dnn_dropout": dnn_dropout,
                "l2_reg": l2_reg,
                "define_new_model": True}

            nn_params = nn_params | hyperparameters
            
            print(f'EXP |{experiment_id}| started on thread {thread_id}')
            best_test_auc, test_auc, best_train_auc, train_auc = 0.0, 0.0, 0.0, 0.0

            for (data, l, d) in data_list:
                for adaptive_loop in range(nn_params["reduce_lr_times"]):
                    if adaptive_loop > 0:
                        nn_params["define_new_model"] = False
                        nn_params['learning_rate_init'] = nn_params['learning_rate_init']*nn_params["reduce_lr_value"]
                        
                    results = train_evalute_model(X={"train":data['X_train'], "test": data['X_test'], "vali": data['X_vali']},
                                    y={'train': data['y_train'], 'test': data['y_test'], "vali": data['y_vali']},
                                    subject=i, 
                                    linear_feature_columns = l[0], 
                                    dnn_feature_columns = d[0], id=experiment_id, **nn_params)
                    
                    best_test_auc += results["BEST_TEST_AUC"]
                    test_auc += results["TEST_AUC"]
                    best_train_auc += results["BEST_TRAIN_AUC"]
                    train_auc += results["TRAIN_AUC"] 
                    count += 1

            mean_best_test_auc = best_test_auc/count   
            mean_test_auc = test_auc/count 
            mean_best_train_auc = best_train_auc/count 
            mean_train_auc = train_auc/count           
            
            finish = time.time()
            experiment_data = {
                "experiment_id": experiment_id, 
                "thread_id": thread_id, 
                "start": datetime.datetime.fromtimestamp(start).strftime('%c'),
                "finish": datetime.datetime.fromtimestamp(finish).strftime('%c'),
                'best_test_auc': mean_best_test_auc,
                'test_auc': mean_test_auc,
                'best_train_auc': mean_best_train_auc,
                'test_auc': mean_train_auc}
            
            results_to_be_saved = experiment_data | nn_params
            global_data_list.append(results_to_be_saved)
        else:
            queue_lock.release()
            time.sleep(1)
    
if __name__ == "__main__":
    create_directiories()
    # data
    subject = '3386'
    bins = 100
    sample_id = 1
    d_mgr = DataManager()
    if sys.argv[1] == None:
        thread_amount = 1
    else:
        thread_amount = int(sys.argv[1])

    experiment_name = f'experiment_{subject}_{time.time()}_threads_{thread_amount}'
    # GET DATA

    data_list = []
    for t in range(OPTIONS['n_datasets']):
        data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, t)
        data_list += [[data, linear_feature_columns, dnn_feature_columns]]
    
    experiments = generate_space([
        # advertiser ids
        # ['1458', '3358', '3386', '3427', '3476', '2259', '2261', '2821', '2997'], 3476, '3386' ~~ problemy
        # smaller advertisers: ['2261', '2259', '2997']
        ['3386'], # '3358', '3476', '2259', '2261', '2821', '2997'
        # '1458', '3358', '3476', '2259', '2261', '2821', '2997'
        # '2821', '2997', '2261', '2259', ?'3476'
        # sample_ids
        list(range(1)),
        # bins
        # [1, 5, 10, 50, 150, 300],
        # alpha
        [0.0001, 0.001], # 0.001, 0.0001, 0.00001, 0.000001
        # lr
        [0.001],
        # batch size
        [1200],
        # hidden
        [tuple(512 for _ in range(4))],
        # dnn dropout
        [0.9, 0.5, 0.3],
        # l2_ref
        [0.00001, 0.0001]
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0) #240(alpha) #126(width), 

    loops = ceil(len(experiments)/thread_amount)
    rest = len(experiments)%thread_amount

    nn_params = {
                    #"batch_size": 1000,
                    # "power_t": 0.5,
                    "max_iter": 250,  # implement
                    # "shuffle": True, # always true
                    "validation_fraction": 0.2,  # implement
                    # "random_state":None,
                    "tol": 1e-5,  # implement OR make sure its low
                    # "warm_start": False,
                    # "momentum": 0.9,
                    # "nesterovs_momentum": True,
                    "early_stopping": True,  # should be always true
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-8,
                    # "n_iter_no_change": 10, "max_fun": 15000
                    "n_iter_no_change": 3,
                    "reduce_lr_times": 3, 
                    "reduce_lr_value": 0.1,
                    "define_new_model": True
                }

    project_start = time.time()
    global_data_list = []

    #define threads
    thread_list = []

    #handle queue
    queue_lock = threading.Lock()
    work_queue = queue.Queue(len(experiments))


    for i in range(thread_amount):
        try:
            thread_list.append(threading.Thread(target=run_learning_thread, args=(data_list, i, nn_params, global_data_list, work_queue)))
        except:
            print(f'unable to create thread {i}')

    #start defined threads
    for id, thread in enumerate(thread_list):
        try:
            thread.start()
        except:
            print('unable to start thread {id} for {loop} loop')

    queue_lock.acquire()
    for items in experiments:
        work_queue.put(items)
    
    queue_lock.release()

    while not work_queue.empty():
        time.sleep(1)

    thread_exit_flag = 1

    for thread_entity in thread_list:
        thread_entity.join()

    project_eval = time.time() - project_start

    print(f'evaluations took: {project_eval}')
    delta_dict = {"global_delta": project_eval}
    pd.DataFrame.from_dict(delta_dict.items()).to_csv(experiment_name)
    df = pd.DataFrame(global_data_list)
    print(df)
    df.to_csv(f'{experiment_name}_ACTUALDATA') 