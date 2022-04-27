import os
import sys
sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np

from torch.nn import BCELoss
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from experiment.ipinyou.onehot.data_manager import DataManager



from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


class EarlyStop:

    def __init__(self, handler, patience=100, max_epochs=None, tol=0):
        self.patience = patience
        self.best_loss = np.inf
        self.failures = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.max_epochs = max_epochs if max_epochs is not None else np.inf
        self.is_best_loss = False
        self.handler = handler
        self.tol = tol
        self.tol_problem = False

    def update_epoch_loss(self, train_loss, validation_loss):
        self.is_best_loss = self.handler.update_epoch_loss(validation_loss)
        self.train_losses.append(train_loss)
        self.val_losses.append(validation_loss)
        self.epoch += 1
        if len(self.val_losses) > 1:
            self.tol_problem = abs(self.val_losses[-2] - self.val_losses[-1]) < self.tol

        if self.is_best_loss:
            self.best_loss = validation_loss
            if self.tol_problem:
                self.failures += 1
            else:
                self.failures = 0
        else:
            self.failures += 1
        print(f'epoch={self.epoch}/{self.max_epochs}, failures={self.failures}/{self.patience}, '
              f'best_loss={self.is_best_loss}, tol={self.tol}, tol_failure={self.tol_problem}')

    def is_best(self):
        return self.is_best_loss

    def is_stop(self):
        return (self.failures > self.patience) or (self.epoch >= self.max_epochs)


class ModelHandler:

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.success_updates = 0

    def reset(self):
        self.best_loss = np.inf
        self.success_updates = 0

    def best(self, path=None):
        path = path if path else self.path
        if path is not None:
            self.model.load_state_dict(torch.load(path + '.best.bin'), strict=True)

    def last(self, path=None):
        path = path if path else self.path
        if path is not None:
            self.model.load_state_dict(torch.load(path + '.last.bin'), strict=True)

    def save(self, path=None, mtype='last'):
        path = path if path else self.path
        if path is not None:
            model_path = path + '.' + mtype + '.bin'
            torch.save(self.model.state_dict(), model_path)

    def update_epoch_loss(self, loss):
        if loss >= self.best_loss:
            return False
        else:
            self.best_loss = loss
            self.success_updates += 1
            return True

def train_loop(model, data, epochs=10, weight_decay=1e-5, patience=5, stabilization=0,
                validation_fraction=0.1, tol=0, epsilon=1e-8, beta_1=0.9, beta_2=0.999, early_stop=True,
                verbose=True, experiment_id=None):
    '''
    
    handler = ModelHandler(model, './model' if experiment_id is None else f'./model_{experiment_id}')
    stop = EarlyStop(patience=patience, max_epochs=epochs, handler=handler, tol=tol)

    while not stop.is_stop(model, data, ):
        pass
    
    '''
    for _ in range(epochs):
        history = model.train_on_batch(x=data['X_train'], y=data['y_train'])
        print(_)

def bce(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def batch_df(df, batch_size):
    df = pd.DataFrame(df)
    batches = list()
    for i in range(int(df.shape[0]/batch_size)):
        batches.append(df.iloc[i*batch_size: i*batch_size+batch_size].to_dict('series'))
    return batches

def batch_y(y, batch_size):
    batches = list()
    for i in range(int(len(y)/batch_size)):
        batches.append(y[i*batch_size:i*batch_size+batch_size])
    return batches


        

if __name__ == "__main__":

    ##  MY OWN CHANGES ##################################################################################
    properties = {"hidden_layer_sizes": tuple(64 for _ in range(4)),
            # "activation":"relu",
            # "solver":'adam',
            "alpha": 0.001,  # 0.000001,
            "batch_size": 1000,
            # "learning_rate": "constant",
            "learning_rate_init": 0.0001,
            # "power_t": 0.5,
            "max_iter": 50,  # implement
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
            "n_iter_no_change": 10}

    device = 'cuda'

    ##  MY OWN CHANGES ##################################################################################
    

    '''
        data = pd.read_csv('./experiment/ipinyou/onehot/criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    
    X = data[data.columns[data.columns!='label']]
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020, stratify=y)
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.2, random_state=2020, stratify=y_train)

    train_model_input = {name: X_train[name] for name in feature_names}
    test_model_input = {name: X_test[name] for name in feature_names}
    vali_model_input = {name: X_vali[name] for name in feature_names}

    print(np.sum(y_train==1), np.sum(y_train==0))

    data = {
        'X_train': train_model_input, 
        'X_test': test_model_input, 
        'X_vali': vali_model_input, 
        'y_train': y_train.values, 
        'y_test': y_test.values, 
        'y_vali': y_vali.values}
    
    '''

    subject = '1458'
    sample_id = 0
    d_mgr = DataManager()
    data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, sample_id)

    # 4.Define Model,
    model = DeepFM(
            linear_feature_columns = linear_feature_columns[0],
            dnn_feature_columns=dnn_feature_columns[0], 
            dnn_hidden_units=properties['hidden_layer_sizes'], 
            task='binary',
            l2_reg_embedding=1e-5, 
            device=device, 
            dnn_dropout=0.9)
    
    optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=properties['learning_rate_init'],
            betas=(properties['beta_1'], properties['beta_2']),
            eps=properties['epsilon'],
            weight_decay=properties['alpha'])

    model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])

    batch_size = 2000
    epochs = 50

    batched_x_vali = batch_df(data['X_vali'], batch_size)
    batched_y_vali = batch_y(data['y_vali'], batch_size)


    weight_decay=1e-5 
    patience=properties['n_iter_no_change']
    stabilization=0
    tol=0 
    early_stop=True
    experiment_id=None

    handler = ModelHandler(model, './model' if experiment_id is None else f'./model_{experiment_id}')
    stop = EarlyStop(patience=patience, max_epochs=epochs, handler=handler, tol=tol)

    epoch = 1
    while not stop.is_stop():
        history = model.fit(data['X_train'], data['y_train'], batch_size=batch_size, verbose=2)
        
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for (x, y) in zip(batched_x_vali, batched_y_vali):
                predictions = model.predict(x)
                loss_value = bce(predictions, y.reshape(-1, 1))
                val_loss += loss_value
                val_steps += 1
        loss = history.history['loss'][-1]
        print(f'EPOCH {epoch}, loss = {loss}, vali_loss = {val_loss/val_steps}')
        epoch += 1

        if stabilization > 0:
            stabilization -= 1
        else:
            stop.update_epoch_loss(validation_loss=np.abs(val_loss / val_steps),
                                   train_loss=np.abs(loss))

            #handler.save(mtype='last')
            if stop.is_best():
                pass
                #handler.save(mtype='best')
    
    pred_ans = model.predict(data['X_test'], batch_size=batch_size)
    print("test AUC", round(roc_auc_score(data['y_test'], pred_ans), 4))
    