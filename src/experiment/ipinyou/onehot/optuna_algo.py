import os
import sys
sys.path.append(os.getcwd())


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
from sklearn.metrics import log_loss, roc_auc_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

VALID_FRACTION = .1

def define_model(trial, linear_feature_columns, dnn_feature_columns):
    # We optimize the number of layers and hidden units in each layer.
    n_layers = trial.suggest_int("n_layers", 4, 8)
    
    layers = []

    for i in range(n_layers):
        out_features =  trial.suggest_int("n_units_l{}".format(i), 8, 32)
        layers.append(out_features)

    dropout = trial.suggest_float("dropout", 0.7, 0.9, log=True)
    l2 = trial.suggest_float("l2", 1e-6, 1e-4, log=True)
    
    return DeepFM(
            linear_feature_columns = linear_feature_columns[0],
            dnn_feature_columns=dnn_feature_columns[0], 
            dnn_hidden_units=layers, 
            task='binary',
            l2_reg_embedding=l2, 
            device=DEVICE, 
            dnn_dropout=dropout)

class HandleDaset(Dataset):
    def __init__(self, x, y):
        device = DEVICE
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


def objective(trial, x_train, y_train, x_test, y_test, linear_feature_columns, dnn_feature_columns):
    

    # Generate the model.
    model = define_model(trial, linear_feature_columns, dnn_feature_columns).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=alpha)


    model.compile(optimizer=optimizer,
             loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])

    # Get the FashionMNIST dataset.
    # tensorize and batch data

    '''
    # train vali split
    validation_fraction = VALID_FRACTION
    X = X_TRAIN
    y = Y_TRAIN

    n = X.shape[0]
    perut = torch.randperm(X.shape[0])
    train_fraction = (1 - validation_fraction)

    X_train_bis = X[perut][0: int(n * train_fraction)]
    y_train_bis = y[perut][0: int(n * train_fraction)]
    X_val = X[perut][int(n * train_fraction):]
    y_val = y[perut][int(n * train_fraction):]
    
    
    train_loader = prep_data(X_train_bis, y_train_bis, batch_size=BATCHSIZE)
    valid_loader = prep_data(X_val, y_val, batch_size=BATCHSIZE) if validation_fraction > 0 else train_loader
    
    ##
    loss_fn = nn.BCELoss()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            #if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
            #    break

            #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target.reshape(-1, 1))
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        val_loss = 0
        val_steps = 0
        valid_sum = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                #if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                #    break
                #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                loss = loss_fn(output, target.reshape(-1, 1))
                local_sum = np.sum((output == target.reshape(-1,1))*1)
                valid_sum += local_sum
                total += output.shape[0]
                val_loss += loss
                val_steps += 1

        avg_loss = val_loss / val_steps
        accuracy = valid_sum/total

        trial.report(accuracy, avg_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    '''
    model.fit(
            x=x_train, 
            y=y_train, 
            batch_size=256, 
            epochs=50, 
            verbose=0,
            validation_data=(x_test, y_test),
            validation_split=0.2)

    

    accuracy = round(roc_auc_score(y_test, model.predict(x_test, 256)), 4)
    trial.report(accuracy)

    return accuracy

def run_optuna(x_train, y_train, x_test, y_test, linear_feature_columns, dnn_feature_columns):
    study = optuna.create_study(
        direction="maximize",
        study_name='deep-fm-study',
        storage='sqlite:///example.db',
        load_if_exists=True)
    study.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, linear_feature_columns, dnn_feature_columns), n_trials=100, timeout=600, show_progress_bar=True)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    # data
    subject = '1458'
    bins = 100
    sample_id = 1

    d_mgr = DataManager()
    X_TRAIN, Y_TRAIN, x_test, y_test, X_train_agge, X_test_agge, linear_feature_columns_list, dnn_feature_columns_list, model_inputs, = d_mgr.get_data(subject, bins, sample_id)

    run_optuna(model_inputs[0], Y_TRAIN, model_inputs[1], y_test, linear_feature_columns_list, dnn_feature_columns_list)