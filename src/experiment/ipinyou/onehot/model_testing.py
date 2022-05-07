import os
from re import sub
import sys
sys.path.append(os.getcwd())

from experiment.ipinyou.onehot.data_manager import DataManager
from experiment.ipinyou.onehot.algo import WDLRunner, DCNRunner, DeepFMRunner, SKLearnMLPRunner, SKLearnLRRunner, MLPRunner, DeepWideRunner
from experiment.measure import ProcessMeasure

import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score


from cgi import test
from re import M
from sklearn.utils import shuffle
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from experiment.display_bis import show_auc
from experiment.ipinyou.onehot.model import DeepWide, define_model, train_model
from experiment.measure import ProcessMeasure
import torch.nn as nn

from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow import keras


subject, sample_id = '1458', 1

d_mgr = DataManager()

data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, sample_id)

X={"train":data['X_train'], "test": data['X_test'], "vali": data['X_vali']}
y={'train': data['y_train'], 'test': data['y_test'], "vali": data['y_vali']}

es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=20, mode='min')
mdckpt = ModelCheckpoint(filepath='model.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('../../files/model.ckpt')
        #model = DeepFM(
        #    linear_feature_columns = linear_feature_columns,
        #    dnn_feature_columns=dnn_feature_columns, 
        #    dnn_hidden_units=properties['hidden_layer_sizes'], 
        #    task='binary',
        #   l2_reg_embedding=1e-5, 
        #    device=device, 
        #   dnn_dropout=0.9)

        
        #model.compile(
        #    optimizer=optimizer,
        #     loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])


        #loss_arr = history.history['loss']
        #print(f'EPOCHS: {len(loss_arr)}')

train_auc = round(roc_auc_score(y['train'], model.predict(X['train'], 256)), 4)
test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], 256)), 4)
print(f"TRAIN_AUC: {train_auc}, TEST_AUC: {test_auc} ")