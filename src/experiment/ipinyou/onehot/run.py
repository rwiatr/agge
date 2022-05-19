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



from deepctr_torch.models import *


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
        # ['1458', '3358', '3386', '3427', '3476', '2259', '2261', '2821', '2997'], 3476, '3386' ~~ problemy
        # smaller advertisers: ['2261', '2259', '2997']
        ['1458'], # '3358', '3476', '2259', '2261', '2821', '2997'
        # '1458', '3358', '3476', '2259', '2261', '2821', '2997'
        # '2821', '2997', '2261', '2259', ?'3476'
        # sample_ids
        list(range(10)),
        # bins
        [150],
        # [1, 5, 10, 50, 150, 300],
        # alpha
        [0.0001], # 0.001, 0.0001, 0.00001, 0.000001
        # lr
        [0.001],
        # hidden
        [tuple(124 for _ in range(4))],
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0) #240(alpha) #126(width), 
    print(experiments)

    sk_auc = []
    torch_auc = []
    elapsed_time = []
    start = time.time()

    prev_subject = None
    df_train, df_test = (None, None)

    sk_learn_mlp = SKLearnMLPRunner().set_measure(measure)
    sk_learn_lr = SKLearnLRRunner().set_measure(measure)
    mlp = MLPRunner().set_measure(measure)
    dw = DeepWideRunner().set_measure(measure)
    deep_fm = DeepFMRunner().set_measure(measure)
    wdl = WDLRunner().set_measure(measure)
    dcn = DCNRunner().set_measure(measure)

    use_bck = False

    d_mgr = DataManager()

    prev_bins = None
    output = "mypc_time_measure"

    for experiment_id, (subject, sample_id, bins, alpha, lr, hidden) in experiments:
        print(f"EXPERIMENT {experiment_id}/{len(experiments) + experiments[0][0]}, data={(subject, sample_id, bins, alpha, lr, hidden)}")
        #X_train, y_train, X_test, y_test, X_train_agge, X_test_agge, linear_feature_columns_list, dnn_feature_columns_list, model_inputs, = d_mgr.get_data(subject, bins, sample_id)
        #linear_feature_columns_list, dnn_feature_columns_list, model_inputs, df_train, df_test, y_train, y_test = d_mgr.get_sparse_dense_data(subject, bins, sample_id)
        data, linear_feature_columns, dnn_feature_columns = d_mgr.get_data_deepfm(subject, sample_id)
        #print(df_train)
        print('----------------------MODEL labels----------------------------')
        #print(model_inputs[0])

        #print(linear_feature_columns_list)

        #print("feature size of agge", X_train_agge.shape[1])
        hidden_sizes = (hidden, 4)

        nn_params = {
            "hidden_layer_sizes": hidden,
            # "activation":"relu",
            # "solver":'adam',
            "alpha": alpha,  # 0.000001,
            "batch_size": 1500,
            # "learning_rate": "constant",
            "learning_rate_init": lr,
            # "power_t": 0.5,
            "max_iter": 500,  # implement
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
            "n_iter_no_change": 500,
            "reduce_lr_times": 1, 
            "reduce_lr_value": 0.1,
            "define_new_model": True
        }

        '''

        deep_fm.run(X={"train":model_inputs[0], "test": model_inputs[1]},
                    y={'train': y_train, 'test': y_test},
                    subject=subject + f";encoding=label;features={len(model_inputs[0])}", 
                    linear_feature_columns = linear_feature_columns_list[0], 
                    dnn_feature_columns = dnn_feature_columns_list[0],  **nn_params)

        dcn.run(X={"train":model_inputs[0], "test": model_inputs[1]},
                    y={'train': y_train, 'test': y_test},
                    subject=subject + f";encoding=label;features={len(model_inputs[0])}", 
                    linear_feature_columns = linear_feature_columns_list[0], 
                    dnn_feature_columns = dnn_feature_columns_list[0],  **nn_params)


        wdl.run(X={"train":model_inputs[0], "test": model_inputs[1]},
                    y={'train': y_train, 'test': y_test},
                    subject=subject + f";encoding=label;features={len(model_inputs[0])}", 
                    linear_feature_columns = linear_feature_columns_list[0], 
                    dnn_feature_columns = dnn_feature_columns_list[0],  **nn_params)


        dw.run({"train": X_train, "test": X_test},
               {"train": y_train, "test": y_test},
               subject + f";encoding=oh;features={X_train.shape[1]}",
               **nn_params)

                mlp.run({"train": X_train, "test": X_test},
                {"train": y_train, "test": y_test},
                subject + f";encoding=oh;features={X_train.shape[1]}",
                **nn_params)

        '''
        start = time.time()
        
        for _ in range(nn_params["reduce_lr_times"]):
            
            print('dcn model evaluation has started!')
            dcn.run(X={"train":data['X_train'], "test": data['X_test'], "vali": data['X_vali']},
                        y={'train': data['y_train'], 'test': data['y_test'], "vali": data['y_vali']},
                        subject=subject + f";encoding=label;features={len(data['X_train'])}", 
                        linear_feature_columns = linear_feature_columns[0], 
                        dnn_feature_columns = dnn_feature_columns[0], **nn_params)
            
            nn_params['learning_rate_init'] = nn_params['learning_rate_init']*nn_params["reduce_lr_value"]
            nn_params["define_new_model"] = False
            if _ == 2:
                nn_params["n_iter_no_change"] = 50
            print(f'time elapsed: {time.time() - start}')
        
        '''
        
        dcn.run(X={"train":model_inputs[0], "test": model_inputs[1]},
                    y={'train': y_train, 'test': y_test},
                    subject=subject + f";encoding=label;features={len(model_inputs[0])}", 
                    linear_feature_columns = linear_feature_columns_list[0], 
                    dnn_feature_columns = dnn_feature_columns_list[0],  **nn_params)


        wdl.run(X={"train":model_inputs[0], "test": model_inputs[1]},
                    y={'train': y_train, 'test': y_test},
                    subject=subject + f";encoding=label;features={len(model_inputs[0])}", 
                    linear_feature_columns = linear_feature_columns_list[0], 
                    dnn_feature_columns = dnn_feature_columns_list[0],  **nn_params)
        '''

        # print(1. / alpha / 1000000)
        # sk_learn_lr.run({"train": X_train, "test": X_test},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=oh;features={X_train.shape[1]}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)


        '''
        
        sk_learn_mlp.run({"train": X_train, "test": X_test},
                          {"train": y_train, "test": y_test},
                          subject + f";encoding=oh;features={X_train.shape[1]}",
                          **nn_params)
        mlp.run({"train": X_train, "test": X_test},
                {"train": y_train, "test": y_test},
                subject + f";encoding=oh;features={X_train.shape[1]}",
                **nn_params)

        dw.run({"train": X_train, "test": X_test},
               {"train": y_train, "test": y_test},
               subject + f";encoding=oh;features={X_train.shape[1]}",
               **nn_params)
        '''

        # sk_learn_lr.run({"train": X_train_agge, "test": X_test_agge},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)
        '''
                sk_learn_mlp.run({"train": X_train_agge, "test": X_test_agge},
                          {"train": y_train, "test": y_test},
                          subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
                          **nn_params)
        mlp.run({"train": X_train_agge, "test": X_test_agge},
                {"train": y_train, "test": y_test},
                subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
                **nn_params)
        dw.run({"train": X_train_agge, "test": X_test_agge},
               {"train": y_train, "test": y_test},
               subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
               **nn_params)

        '''

        if experiment_id % 100 == 0:
            measure.print()

        print(f"writing {output}_{experiment_id % 5}.pickle")
        measure.to_pandas().to_pickle(f"{output}_{experiment_id % 5}.pickle")

    print('-------------------------------- RESULT --------------------------------')
    measure.to_pandas().to_pickle(f"{output}.pickle")
    print(measure.to_pandas())
