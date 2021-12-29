import os, sys

sys.path.append(os.getcwd())


#from experiment.ipinyou.onehot.encoder import MyOneHotEncoder

from random import randrange

import pandas as pd
import numpy as np
import seaborn as sns
from experiment.ipinyou.load import read_data
from experiment.measure import ProcessMeasure
from experiment.ipinyou.onehot.train import train_encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from experiment.display_bis import show_auc
import matplotlib.pyplot as plt
import time
import torch

from scipy.sparse import csr_matrix

from experiment.ipinyou.onehot.model import define_model, train_model

def _generate_space(generators):
    first_generator = generators[0]
    other_generators = generators[1:]

    if len(other_generators) == 0:
        for x in first_generator:
            yield x,
        return

    other = list(_generate_space(other_generators))
    for _first in first_generator:
        for _other in other:
            yield (_first,) + _other


def generate_space(generators, start=0, end=None):
    space = list(enumerate(list(_generate_space(generators))))
    if end is None:
        return space[start:]
    return space[start:end]


def neg_sample(df, ratio):
    clicks = df[df.click == 1]
    not_clicks = df[df.click == 0]
    return pd.concat([clicks, not_clicks.sample(int(df.shape[0] * ratio))], ignore_index=True)



if __name__ == '__main__':
    measure = ProcessMeasure()
    experiments = generate_space([
        # advertiser ids
        # ['1458', '3358', '3386', '3427', '3476', '2259', '2261', '2821', '2997'],
        ['3476'],
        # dims
        [15, 50, 150, 300], # 15, 50, 150, 300
        # re-runs
        list(range(3)),
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0)
    print(experiments)


    auc_results = []
    time_results = []

    prev_subject = None
    df_train, df_test = (None, None)
    for experiment_id, (subject, dims, attempt) in experiments:

        if subject != prev_subject:
            df_train, df_test = read_data(subject)
            df_train.drop(columns=['usertag'], inplace=True)
            df_test.drop(columns=['usertag'], inplace=True)
        prev_subject = subject

        if subject in {'1458', '3386'}:
            _df_test = neg_sample(df_test, 0.2)
            _df_train = neg_sample(df_train, 0.2)
        else:
            _df_test = neg_sample(df_test, 0.5)
            _df_train = neg_sample(df_train, 0.5)

        cols = ['weekday', 'hour',  # 'timestamp',
                 'useragent', 'region', 'city', 'adexchange',
                 'slotwidth', 'slotheight',
                 'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                 'creative',  # 'bidprice', #'payprice',
                 'keypage', 'advertiser']

        #OPTIONS
        hidden_sizes = (7, 7, 7)
        BATCH_SIZE = 500
        PATIENCE = 7
        EPOCHS = 30
        LR = 0.0001

        ## OHE 
        print('ENCODING...')
        enc = OneHotEncoder(handle_unknown='ignore')
        ohe = enc.fit(_df_train[cols])

        X_train = ohe.transform(_df_train[cols])
        y_train = _df_train.click.to_numpy().astype('float64')
        
        X_test = ohe.transform(_df_test[cols])
        y_test = _df_test.click.to_numpy().astype('float64')
        print('ENCODING FINISHED!')

        ## define and train sklearn MLP model
        print('Training sk model')
        start = time.time()
        mlp_sk = MLPClassifier(random_state=0, solver='adam', hidden_layer_sizes=hidden_sizes, batch_size=BATCH_SIZE, validation_fraction=0.1, verbose=1).fit(X_train, y_train)
        elapsed_time_sk = time.time() - start
        print(f'sk model trained in {elapsed_time_sk}')

        ## define and train my MLP model w/o embedding
        mlp = define_model(X_train.shape[1], 1, hidden_sizes, bias=True)
        print('Training my model')
        start =time.time()
        mlp = train_model(model=mlp, X=X_train, y=y_train, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
        elapsed_time_torch = time.time() - start
        print(f'My model trained in {elapsed_time_torch}')

        ## define and train my MLP model with embedding
        mlp_emb = define_model(X_train.shape[1], 1, hidden_sizes, bias=False)
        print('Training my model')
        start = time.time()
        mlp_emb = train_model(model=mlp_emb, X=X_train, y=y_train, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
        elapsed_time_torch_emb = time.time() - start
        print(f'My model trained in {elapsed_time_torch_emb}')


        auc_mlp = show_auc(mlp, X_test, y_test, name=subject)
        auc_sk = show_auc(mlp_sk, X_test, y_test, name=subject)
        auc_emb = show_auc(mlp_emb, X_test, y_test, name=subject)
        
        auc_results += [(auc_mlp, auc_sk, auc_emb)]
        time_results += [(elapsed_time_torch, elapsed_time_sk, elapsed_time_torch_emb)]    
        print('-------------------------------- RESULT --------------------------------')
    
    measure.print()


    ### plotting auc and time 

    plt.figure()
    ax = sns.boxplot(data=pd.DataFrame(auc_results, columns=['torch', 'sk', 'torch_emb']))
    plt.ylabel('auc')
    plt.xlabel('tool')
    plt.show()
    plt.savefig('./model_box_acc.png')

    plt.figure()
    ax = sns.boxplot(data=pd.DataFrame(time_results, columns=['torch', 'sk', 'torch_emb']))
    plt.ylabel('training time')
    plt.xlabel('Tool')
    plt.show()
    plt.savefig('./model_box_time.png')

    '''
    
        X_train = _df_train[['weekday', 'hour',  # 'timestamp',
                 'useragent', 'region', 'city', 'adexchange',
                 'slotwidth', 'slotheight',
                 'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                 'creative',  # 'bidprice', #'payprice',
                 'keypage', 'advertiser']].to
        
        y_train = _df_train.click.astype('float64')


        X_test = _df_test[['weekday', 'hour',  # 'timestamp',
                 'useragent', 'region', 'city', 'adexchange',
                 'slotwidth', 'slotheight',
                 'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                 'creative',  # 'bidprice', #'payprice',
                 'keypage', 'advertiser']].astype('float64')

        y_test = _df_test.click.astype('float64')

        
        
        
        hfe = train_encoder(_df_train, size=dims)
        X_train = hfe.transform(_df_train).astype('float64')
        y_train = _df_train.click.to_numpy().astype('float64')
        X_test = hfe.transform(_df_test).astype('float64')
        y_test = _df_test.click.to_numpy().astype('float64')

        measure.set_suffix('_1_None_f={}_b=-1_bt=-1'.format(X_train.shape[1]))
        measure.start(subject)
        lr = LogisticRegression(random_state=0, max_iter=10000, verbose=0, solver='lbfgs').fit(X_train, y_train)
        auc = show_auc(lr, X_test, y_test, name=subject)
        measure.data_point(auc, collection='auc_{}'.format(subject))
        measure.stop(subject)
        measure.print()
        print('Done experiment id={}, adv={}, dims={}, attempt={}'.format(experiment_id, subject, dims, attempt))


        w poszukiwaniu optymalnych parametrów deterministycznego modelu wzrostu wartości przedsiębiorstwa
    '''

    
