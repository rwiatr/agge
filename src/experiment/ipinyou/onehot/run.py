import os, sys

sys.path.append(os.getcwd())


#from experiment.ipinyou.onehot.encoder import MyOneHotEncoder

from random import randrange

import pandas as pd
import numpy as np
from experiment.ipinyou.load import read_data
from experiment.measure import ProcessMeasure
from experiment.ipinyou.onehot.train import train_encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from experiment.display_bis import show_auc

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
        ['3427'],
        # dims
        [15, 50, 150, 300],
        # re-runs
        list(range(3)),
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0)
    print(experiments)


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

        enc = OneHotEncoder(handle_unknown='ignore')
        ohe = enc.fit(_df_train[cols])

        X_train = ohe.transform(_df_train[cols])
        y_train = _df_train.click.to_numpy().astype('float64')
        
        X_test = ohe.transform(_df_test[cols])
        y_test = _df_test.click.to_numpy().astype('float64')

        print(X_train.shape, X_test.shape)
        
        measure.set_suffix('_1_None_f={}_b=-1_bt=-1'.format(X_train.shape[1]))
        measure.start(subject)


        #mlp = LogisticRegression(random_state=0, max_iter=10000, verbose=0, solver='lbfgs').fit(X_train, y_train)
        
        #mlp = MLPClassifier(random_state=0, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,50)).fit(X_train, y_train)
        mlp = define_model(X_train.shape[1], 1, (400, 100, 100, 25))
        train_model(model=mlp, X=X_train, y=y_train, lr=0.0001, epochs=50, batch_size=1000)
        
        auc = show_auc(mlp, X_test, y_test, name=subject)
        measure.data_point(auc, collection='auc_{}'.format(subject))
        measure.stop(subject)
        measure.print()
        print('Done experiment id={}, adv={}, dims={}, attempt={}'.format(experiment_id, subject, dims, attempt))
        print(f'The result is: {auc}')
    print('-------------------------------- RESULT --------------------------------')
    measure.print()

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

    
