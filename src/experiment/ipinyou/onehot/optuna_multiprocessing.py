import os
import sys
from zoneinfo import available_timezones
sys.path.append(os.getcwd())
import threading

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader
from experiment.ipinyou.onehot.data_manager import DataManager
from experiment.ipinyou.onehot.model import Mlp
from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss, roc_auc_score
import time
import pandas as pd
from optuna.samplers import TPESampler, CmaEsSampler
import copy
import torch.multiprocessing as mp

OPTIONS = {
    'n_trials': 10,
    'n_datasets': 2,
    'batch_size': 1200,
    'patience': 3,
    'epochs': 8,
    'adaptive_lr_depth': 4,
    'adaptive_lr_init': 0.01,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'sampler': TPESampler, # TPESampler(), # CmaEsSampler()
    'model': DeepFM, # DeepFM, #WDL #DCN
    'subject': '3358'
}

def create_directiories(dirs = [f'./optuna_data/threads', './models_optuna/', f'./optuna_data/global']):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f'directory {dir} has been created')
        else:
            print(f'directory {dir} already exists')


def define_model(trial, linear_feature_columns, dnn_feature_columns):
    # We optimize the number of layers and hidden units in each layer.
    n_layers = trial.suggest_int("n_layers", 3, 5)
    out_features =  trial.suggest_int("n_units_l", 200, 500)

    layers = [out_features for _ in range(n_layers)]

    dropout = trial.suggest_float("dropout", 0.1, 0.9, log=True)
    l2 = trial.suggest_float("l2", 1e-6, 1e-4, log=True)
    
    return OPTIONS['model'](
            linear_feature_columns = linear_feature_columns[0],
            dnn_feature_columns=dnn_feature_columns[0], 
            dnn_hidden_units=layers, 
            task='binary',
            l2_reg_embedding=l2, 
            device=OPTIONS['DEVICE'], 
            dnn_dropout=dropout)

class HandleDaset(Dataset):
    def __init__(self, x, y):
        device = OPTIONS['DEVICE']
        self.x = self.sparse_to_tensor(x).to(device)
        # self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

    def spy_sparse2torch_sparse(self, data):
        """

        :param data: a scipy sparse csr matrix
        :return: a sparse torch tensor
        
        """
        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
        t = torch.sparse_coo_tensor(data)
        return t

    def sparse_to_tensor(self, sparse_m):
        if type(sparse_m) is np.ndarray:
            return torch.from_numpy(sparse_m).float()
        sparse_m = sparse_m.tocoo()

        values = sparse_m.data
        indices = np.vstack((sparse_m.row, sparse_m.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = sparse_m.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def prep_data(X, y, batch_size, shuffle=False):
    return DataLoader(HandleDaset(x=X, y=y), batch_size=batch_size, shuffle=shuffle, drop_last=True)


def objective(trial, data_list, device):

    #data_list = copy.deepcopy(data_list)
    linear_feature_columns, dnn_feature_columns = data_list[0][1], data_list[0][2]

    device = torch.device('cuda:{device}')
    model = define_model(trial, linear_feature_columns, dnn_feature_columns).to(device)
    #print(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
    
    lr = OPTIONS['adaptive_lr_init']
    mean_auc = 0.0

    for (id, (data, lfc, dfc)) in enumerate(data_list):
        mdckpt = ModelCheckpoint(filepath=f'./models_optuna/model_cmd{trial.number}_{id}.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
        #es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=OPTIONS['patience'], mode='min')

        for i in range(OPTIONS['adaptive_lr_depth']):
            if i != 0:
                model = torch.load(f'./models_optuna/model_cmd{trial.number}_{id}.ckpt')
                
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=alpha)
            model.compile(optimizer=optimizer,
                    loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])

            model.fit(
                    x=data['X_train'], 
                    y=data['y_train'], 
                    batch_size=OPTIONS['batch_size'], 
                    epochs=OPTIONS['epochs'], 
                    verbose=0,
                    validation_data=(data['X_vali'], data['y_vali']),
                    shuffle=False,
                    callbacks=[mdckpt])

            model_best = torch.load(f'./models_optuna/model_cmd{trial.number}_{id}.ckpt')
            auc = round(roc_auc_score(data['y_test'], model_best.predict(data['X_test'], 256)), 4)
            lr = lr * 0.1
        mean_auc += auc
    mean_auc = mean_auc/len(data_list)

    return mean_auc

def run_optuna(data_list, study_name, device):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage='sqlite:///deepfm_study.db',
        load_if_exists=True,
        sampler=OPTIONS['sampler']())
    
    study.optimize(lambda trial: objective(trial, data_list, device), n_trials=OPTIONS['n_trials'], n_jobs=-1, timeout=None, show_progress_bar=False)


if __name__ == "__main__":
    # data
    subject = OPTIONS['subject']
    model = OPTIONS['model'].__name__
    sampler = OPTIONS['sampler'].__name__
    bins = 100
    sample_id = 1
    
    create_directiories()
    d_mgr = DataManager()
    available_gpus = torch.cuda.device_count()

    study_name = f'{model}_{subject}_{sampler}_{str(int(time.time()))}_processes_{available_gpus}'

    # GET DATA
    data_list = []
    for t in range(OPTIONS['n_datasets']):
        data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, t)
        data_list += [[data, linear_feature_columns, dnn_feature_columns]]

    main_study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage='sqlite:///deepfm_study.db',
        load_if_exists=False)

    available_gpus = torch.cuda.device_count()
    devices = []
    for cuda in range(available_gpus):
        devices.append(cuda)

    processes = []
    for device in devices:
        p = mp.Process(target=run_optuna, args=(data_list, study_name, device)) 
        p.start() 
        processes.append(p)

    for p in processes:
        p.join()  
    
    print(main_study.trials_dataframe())
    main_study.trials_dataframe().to_csv(f'./optuna_data/global/OPTUNAstudy_{study_name}_processes_{available_gpus}.csv')
    print('EVERYTHING HAS FINISHED')
