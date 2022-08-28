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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import time
import pandas as pd
from optuna.samplers import TPESampler, CmaEsSampler
import copy

OPTIONS = {
    'n_trials': 50,
    'n_datasets': 3,
    'batch_size': 300,
    'patience': 3,
    'epochs': 300,
    'adaptive_lr_depth': 2,
    'adaptive_lr_init': 0.01,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'sampler': CmaEsSampler, # TPESampler(), # CmaEsSampler()
    'model': LogisticRegression, # DeepFM, #WDL #DCN
    'subject': '2259'
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


def objective(trial, data_list):

    solver = trial.suggest_categorical("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
    max_iter = trial.suggest_int("max_iter", 500, 20000)

    mean_auc = 0.0

    for (id, (X_train_agge, y_train, X_test_agge, y_test)) in enumerate(data_list):
        
        lr = LogisticRegression(max_iter=max_iter, verbose=1, solver=solver, C=1. / alpha / 1000000).fit(X_train_agge, y_train)
        y_pred_proba = lr.predict_proba(X_test_agge)[::,1]
        auc = roc_auc_score(y_test, y_pred_proba)

        mean_auc += auc
    mean_auc = mean_auc/len(data_list)

    return mean_auc

def run_optuna(data_list, study_name, njobs):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage='sqlite:///deepfm_study.db',
        load_if_exists=True,
        sampler=OPTIONS['sampler']())
    
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
    study.trials_dataframe().to_csv(f'./optuna_data/global/OPTUNAstudy_{study_name}_threads_{njobs}.csv')

if __name__ == "__main__":
    # data
    subject = OPTIONS['subject']
    model = OPTIONS['model'].__name__
    sampler = OPTIONS['sampler'].__name__
    bins = 100
    sample_id = 1
    
    create_directiories()

    d_mgr = DataManager()
    if sys.argv[1] == None:
        thread_amount = 1
    else:
        thread_amount = int(sys.argv[1])

    study_name = f'{model}_{subject}_{sampler}_{str(int(time.time()))}_threads_{thread_amount}'

    # GET DATA
    data_list = []
    for t in range(OPTIONS['n_datasets']):
        X_train, y_train, X_test, y_test, X_train_agge, X_test_agge, linear_feature_columns_list, dnn_feature_columns_list, model_inputs, = d_mgr.get_data(subject, bins, sample_id)
        data_list += [[X_train_agge, y_train, X_test_agge, y_test]]

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
