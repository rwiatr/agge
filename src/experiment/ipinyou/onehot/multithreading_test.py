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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import copy

def create_directiories(dirs = [f'./mt_data/threads_{sys.argv[1]}', './models_mt']):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f'directory {dir} has been created')
        else:
            print(f'directory {dir} already exists')

def train_evalute_model(subject, X, y , linear_feature_columns, dnn_feature_columns, id, **properties):
    
    es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=properties["n_iter_no_change"], mode='min')
    mdckpt = ModelCheckpoint(filepath=f'./models_mt/model_id{id}.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if properties['define_new_model']:
        model = DeepFM(
            linear_feature_columns = linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns, 
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
    print(f'Model evaluation for lr={current_lr_value}')

    history = model.fit(
        x=X['train'], 
        y=y['train'], 
        batch_size=properties['batch_size'], 
        epochs=properties['max_iter'], 
        verbose=2,
        validation_data=(X['vali'], y['vali']),
        shuffle=False,
        callbacks=[es, mdckpt])

    loss_arr = history.history['loss']
    print(f'EPOCHS: {len(loss_arr)}')
    model_best = torch.load(f'./models_mt/model_id{id}.ckpt')

    train_auc = round(roc_auc_score(y['train'], model.predict(X['train'], properties['batch_size'])), 4)
    test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], properties['batch_size'])), 4)
    best_train_auc = round(roc_auc_score(y['train'], model_best.predict(X['train'], properties['batch_size'])), 4)
    best_test_auc = round(roc_auc_score(y['test'], model_best.predict(X['test'], properties['batch_size'])), 4)
    print(f"TRAIN_AUC: {train_auc}, TEST_AUC: {test_auc}, BEST_TEST_AUC: {best_test_auc} ")
    results = {"TRAIN_AUC": train_auc, "TEST_AUC": test_auc, "BEST_TEST_AUC": best_test_auc, "BEST_TRAIN_AUC": best_train_auc}
    return results

    

def run_learning_thread(data, l, d, i, nn_params):
    start = time.time()
    nn_params = copy.deepcopy(nn_params)
    
    for _ in range(nn_params["reduce_lr_times"]):
        if _ > 0:
            nn_params["define_new_model"] = False
            nn_params['learning_rate_init'] = nn_params['learning_rate_init']*nn_params["reduce_lr_value"]
            
        results = train_evalute_model(X={"train":data['X_train'], "test": data['X_test'], "vali": data['X_vali']},
                        y={'train': data['y_train'], 'test': data['y_test'], "vali": data['y_vali']},
                        subject=i, 
                        linear_feature_columns = l[0], 
                        dnn_feature_columns = d[0], id=i, **nn_params)
                    
    save_data = results | nn_params
    
    total_time = time.time() - start
    print(f'time elapsed: {total_time}')
    save_data['delta'] = total_time
    pd.DataFrame.from_dict(save_data.items()).to_csv(f'./mt_data/threads_{sys.argv[1]}/{time.time()}_thread{id}.csv')
    
if __name__ == "__main__":
    create_directiories()
    # data
    subject = '2261'
    bins = 100
    sample_id = 1
    d_mgr = DataManager()
    thread_amount = int(sys.argv[1])
    experiment_name = f'experiment_{subject}_{time.time()}_threads{thread_amount}'
    # GET DATA
    data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, sample_id)
    
    experiments = generate_space([
        # advertiser ids
        # ['1458', '3358', '3386', '3427', '3476', '2259', '2261', '2821', '2997'], 3476, '3386' ~~ problemy
        # smaller advertisers: ['2261', '2259', '2997']
        ['2261'], # '3358', '3476', '2259', '2261', '2821', '2997'
        # '1458', '3358', '3476', '2259', '2261', '2821', '2997'
        # '2821', '2997', '2261', '2259', ?'3476'
        # sample_ids
        list(range(5)),
        # bins
        # [1, 5, 10, 50, 150, 300],
        # alpha
        [0.0001], # 0.001, 0.0001, 0.00001, 0.000001
        # lr
        [0.001],
        # hidden
        [tuple(256 for _ in range(4)), tuple(512 for _ in range(4)), tuple(256 for _ in range(8)), tuple(512 for _ in range(8))],
        # dnn dropout
        [0.9],
        # l2_ref
        [0.00001]
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0) #240(alpha) #126(width), 

    print(experiments)
    print(len(experiments))
    print(ceil(len(experiments)/thread_amount))

    loops = ceil(len(experiments)/thread_amount)
    rest = len(experiments)%thread_amount

    nn_params = {
                    "batch_size": 1600,
                    # "power_t": 0.5,
                    "max_iter": 2000,  # implement
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
                    "n_iter_no_change": 15,
                    "reduce_lr_times": 4, 
                    "reduce_lr_value": 0.1,
                    "define_new_model": True
                }
    project_start = time.time()
    for loop in range(loops):
        if loop != loops-1:
            thread_count = thread_amount
            #print(experiments[loop*thread_amount+thread])   
        else:
            thread_count = thread_amount if rest == 0 else rest
            #print(experiments[loop*thread_amount+thread])
        print(f'LLOP: {loop}')
        #define threads
        thread_list = []
        for i in range(thread_count):
            try:
                experiment_id, (subject, sample_id, alpha, lr, hidden, dnn_dropout,l2_reg) = experiments[loop*thread_amount+i]
                hyperparameters = {
                    "hidden_layer_sizes": hidden,
                    "alpha": alpha,
                    "learning_rate_init": lr,
                    "dnn_dropout": dnn_dropout,
                    "l2_reg": l2_reg}
                params = nn_params | hyperparameters
                thread_list += [threading.Thread(target=run_learning_thread, args=(data, linear_feature_columns, dnn_feature_columns, experiment_id, params))]
            except:
                print(f'unable to create thread {experiment_id}')

        #start defined threads
        for id, thread in enumerate(thread_list):
            try:
                thread.start()
            except:
                print('unable to start thread {id} for {loop} loop')
        
        for thread_entity in thread_list:
            thread_entity.join()

        project_eval = time.time() - project_start

        print(f'evaluations took: {project_eval}')
        delta_dict = {"global_delta": project_eval}
        pd.DataFrame.from_dict(delta_dict.items()).to_csv(experiment_name)


'''
for experiment_id, (sample_id, alpha, lr, hidden, dnn_dropout, l2_reg) in experiments:
        print(f"EXPERIMENT {experiment_id}/{len(experiments) + experiments[0][0]}, data={(sample_id, alpha, lr, hidden, dnn_dropout, l2_reg)}")


        #define threads
        thread_list = []
        for i in range(int(sys.argv[1])):
            try:
                thread_list += [threading.Thread(target=run_learning_thread, args=(data, linear_feature_columns, dnn_feature_columns, i))]
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
        #run_learning_thread(data, linear_feature_columns, dnn_feature_columns)

'''
    