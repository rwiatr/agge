import os
import sys

from experiment.ipinyou.onehot.data_manager import DataManager

sys.path.append(os.getcwd())

from experiment.ipinyou.onehot.algo import SKLearnMLPRunner, SKLearnLRRunner, MLPRunner, DeepWideRunner

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
        ['2261', '2821', '2997'],
        # sample_ids
        list(range(15)),
        # bins
        [150],
        # [1, 5, 10, 50, 150, 300],
        # alpha
        [0.001, 0.01, 0.1],
        # hidden
        [32],
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0)
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
    use_bck = False

    d_mgr = DataManager()

    prev_bins = None
    output = "result__10"

    for experiment_id, (subject, sample_id, bins, alpha, hidden) in experiments:
        print(f"EXPERIMENT {experiment_id}/{len(experiments)}, data={(subject, sample_id, bins, alpha, hidden)}")
        X_train, y_train, X_test, y_test, X_train_agge, X_test_agge = d_mgr.get_data(subject, bins, sample_id)

        print("feature size of agge", X_train_agge.shape[1])
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
            "n_iter_no_change": 5
        }

        # print(1. / alpha / 1000000)
        # sk_learn_lr.run({"train": X_train, "test": X_test},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=oh;features={X_train.shape[1]}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)
        # sk_learn_mlp.run({"train": X_train, "test": X_test},
        #                  {"train": y_train, "test": y_test},
        #                  subject + f";encoding=oh;features={X_train.shape[1]}",
        #                  **nn_params)
        mlp.run({"train": X_train, "test": X_test},
                {"train": y_train, "test": y_test},
                subject + f";encoding=oh;features={X_train.shape[1]}",
                **nn_params)
        dw.run({"train": X_train, "test": X_test},
               {"train": y_train, "test": y_test},
               subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
               **nn_params)
        # sk_learn_lr.run({"train": X_train_agge, "test": X_test_agge},
        #                 {"train": y_train, "test": y_test},
        #                 subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
        #                 random_state=0, max_iter=10000, verbose=1, solver='lbfgs', C=1. / alpha / 1000000)
        # sk_learn_mlp.run({"train": X_train_agge, "test": X_test_agge},
        #                  {"train": y_train, "test": y_test},
        #                  subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
        #                  **nn_params)
        mlp.run({"train": X_train_agge, "test": X_test_agge},
                {"train": y_train, "test": y_test},
                subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
                **nn_params)
        dw.run({"train": X_train_agge, "test": X_test_agge},
               {"train": y_train, "test": y_test},
               subject + f";encoding=agge;features={X_train_agge.shape[1]};bins={bins}",
               **nn_params)

        if experiment_id % 100 == 0:
            measure.print()

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
