from random import randrange

import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiment.display import show_auc
from experiment.ipinyou.hash.train import train_encoder
from experiment.ipinyou.load import read_data
from experiment.measure import ProcessMeasure


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
        ['2997'],
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

            for column in df_train.columns:
                print(column)
                df_train[column] = df_train[column].astype('category')
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
            print(len(df_train.columns))

        prev_subject = subject

        if subject in {'1458', '3386'}:
            _df_test = neg_sample(df_test, 0.2)
            _df_train = neg_sample(df_train, 0.2)
        else:
            _df_test = neg_sample(df_test, 0.5)
            _df_train = neg_sample(df_train, 0.5)
        dims = dims + randrange(11) - 5

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

    print('-------------------------------- RESULT --------------------------------')
    measure.print()