import os
import sys
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

OPTIONS = {
    'n_trials': 10,
    'n_datasets': 3,
    'batch_size': 1200,
    'patience': 3,
    'epochs': 1,
    'adaptive_lr_depth': 2,
    'adaptive_lr_init': 0.01,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'sampler': TPESampler() # CmaEsSampler()
}

def create_directiories(dirs = [f'./optuna_data/threads_{sys.argv[1]}', './models_optuna/', f'./optuna_data/global']):
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
    
    return DeepFM(
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


def objective(trial, data_list):

    data_list = copy.deepcopy(data_list)
    linear_feature_columns, dnn_feature_columns = data_list[0][1], data_list[0][2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(trial, linear_feature_columns, dnn_feature_columns).to(device)
    print(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
    
    lr = OPTIONS['adaptive_lr_init']
    mean_auc = 0.0

    for (id, (data, lfc, dfc)) in enumerate(data_list):
        mdckpt = ModelCheckpoint(filepath=f'./models_optuna/model_cmd{trial.number}_{id}.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=OPTIONS['patience'], mode='min')

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
                    callbacks=[es, mdckpt])

            model_best = torch.load(f'./models_optuna/model_cmd{trial.number}_{id}.ckpt')
            auc = round(roc_auc_score(data['y_test'], model_best.predict(data['X_test'], 256)), 4)
            lr = lr * 0.1
        mean_auc += auc
    mean_auc = mean_auc/len(data_list)

    return mean_auc

def run_optuna(data_list, study_name, njobs):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage='sqlite:///deepfm_study.db',
        load_if_exists=True,
        sampler=OPTIONS['sampler'])
    
    study.optimize(lambda trial: objective(trial, data_list), n_trials=OPTIONS['n_trials'], n_jobs=njobs, timeout=None, show_progress_bar=False)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    total_time = time.time() - start
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    print(trial)

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    params_dict = dict(trial.params.items())
    print(f'EXPERIMENT RUN FOR {total_time} s')
    params_dict['delta'] = total_time
    params_dict['study_name'] = study_name
    params_dict['value'] = trial.value
    params_dict['finished_trials'] = len(study.trials)
    params_dict['complete_trials'] = len(complete_trials)
    pd.DataFrame.from_dict(params_dict.items()).to_csv(f'./optuna_data/threads_{njobs}/study_{study_name}_{time.time()}_threads{njobs}.csv')

    print(study.trials_dataframe())
    study.trials_dataframe().to_csv(f'./optuna_data/global/DFstudy_{study_name}_{time.time()}_thread_{njobs}.csv')

if __name__ == "__main__":
    # data
    subject = '3358'
    bins = 100
    sample_id = 1
    
    create_directiories()

    d_mgr = DataManager()
    if sys.argv[1] == None:
        thread_amount = 1
    else:
        thread_amount = int(sys.argv[1])

    study_name = f'deepfm_{subject}_{str(int(time.time()))}_threads{thread_amount}'

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

    start = time.time()
    run_optuna(data_list, study_name, thread_amount)

    finish = time.time() - start
    time_dict = {'delta':finish}
    print(finish)
    #pd.DataFrame.from_dict(time_dict.items()).to_csv(f'./optuna_data/global/DELTA_{study_name}_threads{int(sys.argv[1])}_{time.time()}')
    print('EVERYTHING HAS FINISHED')
