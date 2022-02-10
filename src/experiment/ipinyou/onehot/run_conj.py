import os
import sys

from experiment.ipinyou.onehot.data_manager2 import DataManager, datasets, datasets_onehot_conj

sys.path.append(os.getcwd())

from experiment.ipinyou.onehot.algo import SKLearnMLPRunner, SKLearnLRRunner, MLPRunner, DeepWideRunner, \
    DeepWideRunnerV2

import pandas as pd
from experiment.measure import ProcessMeasure
import matplotlib.pyplot as plt
import time


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
        ['3386'], # 1458
        # ['2261', '2821', '2997'], # conj_0 and 1
        # ['1458', '3358', '3386'],
        # sample_ids
        list(range(10)),
        # bins
        # [50, 150, 300],
        ["N/A"],
        # [1, 5, 10, 50, 150, 300],
        # alpha
        [0.01, 0.001, 0.0001],
        # 2821 - [0.01, 0.001 (best) -> C=0.01, 0.0001] - bins 35 - features 4808
        # 2261 - [0.01 (best-agge) -> C=0.001, 0.001, 0.0001] analyze - bins 30 - features - 4704, 35 - 5190, 32 - 4916, 31 - 4804
        # 2997 - [0.01, 0.001 (best) -> C=0.01, 0.0001] - bins 31 - 3485, bins - 54
        # hidden
        [32],
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=27)
    print(experiments)

    sk_auc = []
    torch_auc = []
    elapsed_time = []
    start = time.time()

    prev_subject = None
    df_train, df_test = (None, None)

    output = "conj_7b"
    # output = "conj_2821_alt_long_iter"

    sk_learn_mlp = SKLearnMLPRunner().set_measure(measure)
    sk_learn_lr = SKLearnLRRunner().set_measure(measure)
    mlp = MLPRunner(output).set_measure(measure)
    dw = DeepWideRunner(output).set_measure(measure)
    dwv2 = DeepWideRunnerV2(output).set_measure(measure)
    use_bck = False

    d_mgr = DataManager()

    prev_bins = None
    conj = True
    id_base = 0
    for experiment_id, (subject, sample_id, bins, alpha, hidden) in experiments:
        sample_id += id_base
        bins = {
            '2261': 31,
            '2821': 35,
            '2997': 54,
            '3427': 29,

            '3358': 32,
            '3476': 30,
            '3386': 33
        }[subject]

        exp = f"EXPERIMENT {experiment_id}/{len(experiments)}, " \
              f"data={(subject, sample_id, bins, alpha, hidden)}, file={output}"
        print(exp)
        os.system(f"echo '{exp}' >> exp.log")
        ohe_conj_ht = None
        ohe_conj_agge = None
        X, y = d_mgr.get_data(subject, bins, sample_id, conj=conj, agge=True, agge_conj=True)
        ohe_dataset = datasets(X, y)
        agge_ds = datasets(X, y, agge=True)
        ohe_conj_ht = datasets_onehot_conj(X, y, agge=False)
        ohe_conj_agge = datasets_onehot_conj(X, y, agge=True)
        X = y = None
        #
        # print("feature size of agge", agge_ds[0]['train'].shape[1], agge_ds[0]['train_conj'].shape[1],
        #       agge_ds[0]['train_all'].shape[1])
        # print("feature size of ohe", ohe_dataset[0]['train'].shape[1], ohe_dataset[0]['train_conj'].shape[1],
        #       ohe_dataset[0]['train_and_conj'].shape[1])
        hidden_sizes = (hidden, 4)

        nn_params = {
            "hidden_layer_sizes": hidden_sizes,
            # "activation":"relu",
            # "solver":'adam',
            "alpha": alpha,  # 0.000001,
            "batch_size": 1000,
            # "learning_rate": "constant",
            "learning_rate_init": 0.0001,
            # "power_t": 0.5,
            "max_iter": 50,  # implement
            # "shuffle": True, # always true
            "validation_fraction": 0.0,  # implement
            # "random_state":None,
            "tol": 1e-4,  # implement OR make sure its low
            # "warm_start": False,
            # "momentum": 0.9,
            # "nesterovs_momentum": True,
            "early_stopping": False,  # should be always true
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            # "n_iter_no_change": 10, "max_fun": 15000
            "n_iter_no_change": 5
        }
        C = 1. / alpha / 100000
        print(C)
        # print(1. / alpha / 1000000)
        # sk_learn_lr.run({"train": X_train, "test": X_test},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=oh;features={X_train.shape[1]}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)
        # sk_learn_mlp.run({"train": X_train, "test": X_test},
        #                  {"train": y_train, "test": y_test},
        #                  subject + f";encoding=oh;features={X_train.shape[1]}",
        #                  **nn_params)
        print("sk_learn_lr onehot")
        print(f"sk_learn_lr onehot dims=({ohe_dataset[0]['train'].shape},"
              f" {ohe_dataset[0]['test'].shape})")
        solver = 'saga'
        # solver = 'lbfgs' #saga
        maxiter = 40
        sk_learn_lr.run(*ohe_dataset,
                        subject + f";encoding=oh;features={ohe_dataset[0]['train'].shape[1]};"
                        # f"features_conj={ohe_dataset[0]['train_conj'].shape[1]};"
                                  f"conj=False",
                        max_iter=maxiter, verbose=1, solver=solver, C=C)
        print(f"sk_learn_lr agge dims=({agge_ds[0]['train'].shape},"
              f" {agge_ds[0]['test'].shape})")
        sk_learn_lr.run(*agge_ds,
                        subject + f";encoding=agge;features={agge_ds[0]['train'].shape[1]};"
                        # f"features_conj={agge_ds[0]['train_conj'].shape[1]};"
                                  f"conj=False;bins={bins}",
                        max_iter=maxiter, verbose=1, solver=solver, C=C)

        print(f"sk_learn_lr onehot + ht_conj dims=({ohe_conj_ht[0]['train'].shape},"
              f" {ohe_conj_ht[0]['test'].shape})")
        sk_learn_lr.run(*ohe_conj_ht,
                        subject + f";encoding=oh+ht;features="
                                  f"{ohe_conj_ht[0]['train'].shape[1]};"
                                  f"conj=True",
                        max_iter=maxiter, verbose=1, solver=solver, C=C)
        print(f"sk_learn_lr onehot + agge_conj dims=({ohe_conj_ht[0]['train'].shape},"
              f" {ohe_conj_ht[0]['test'].shape})")
        # measure.print()

        sk_learn_lr.run(*ohe_conj_agge,
                        subject + f";encoding=oh+agge;features="
                                  f"{ohe_conj_agge[0]['train'].shape[1]};"
                                  f"conj=True;bins={bins}",
                        max_iter=maxiter, verbose=1, solver=solver, C=C)
        # mlp.run(*ohe_dataset, subject + f";encoding=oh;features={ohe_dataset[0]['train'].shape[1]}", **nn_params)
        # dwv2.run(*ohe_dataset,
        #          subject + f";encoding=oh;features={ohe_dataset[0]['train'].shape[1]};"
        #                    f"features_conj={ohe_dataset[0]['train_conj'].shape[1]};conj={conj};bins={bins}",
        #          **nn_params)
        # dw.run(*ohe_dataset, subject + f";encoding=oh;features={len(cols)};bins={bins}", **nn_params)
        # sk_learn_lr.run({"train": X_train_agge, "test": X_test_agge},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)
        # sk_learn_mlp.run({"train": X_train_agge, "test": X_test_agge},
        #                  {"train": y_train, "test": y_test},
        #                  subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
        #                  **nn_params)
        # mlp.run(*agge_ds, subject + f";encoding=agge;features={X_train_agge[cols].shape[1]};bins={bins}", **nn_params)
        # dw.run(*agge_ds, subject + f";encoding=agge;features={X_train_agge[cols].shape[1]};bins={bins}", **nn_params)
        measure.print()
        # if experiment_id % 100 == 0:
        #     measure.print()

        print(f"writing {output}_{experiment_id % 5}.pickle")
        measure.to_pandas().to_pickle(f"{output}_{experiment_id % 5}.pickle")

    print('-------------------------------- RESULT --------------------------------')
    measure.to_pandas().to_pickle(f"{output}.pickle")
    print(measure.to_pandas())
    plt.figure()
    plt.plot(sk_auc, label='sk')
    plt.ylabel('auc')
    plt.xlabel('model #')
    plt.plot(torch_auc, label='torch')

    plt.legend()
    plt.show()
    plt.savefig('./model_acc.png')

    plt.figure()
    plt.plot(elapsed_time, label=['sk', 'torch'])
    plt.ylabel('training time')
    plt.xlabel('model #')
    plt.legend()
    plt.show()
    plt.savefig('./model_time.png')
    plt.savefig('./model_time.png')
