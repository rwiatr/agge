import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment.display import show_auc
from estimators import UnaryEstimator
from experiment.measure import ProcessMeasure


def results_f(lr, X_train, y_train, X_test, y_test, subject):
    show_auc(lr, X_train, y_train, name=subject)
    return show_auc(lr, X_test, y_test, name=subject)


def train_estimators(df_train, df_test, msk, normalize_column='minmax', bins=None, bin_type='cnt'):
    estimator_columns = ['weekday', 'hour',  # 'timestamp',
                         'useragent', 'region', 'city', 'adexchange',
                         'slotwidth', 'slotheight',
                         'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                         'creative',  # 'bidprice', #'payprice',
                         'keypage', 'advertiser']

    msk = msk if msk is not None else np.ones(df_train.shape[0]).astype(bool)
    unary_estimators = []
    for c in estimator_columns:
        unary_estimators.append(UnaryEstimator(value_column='click', key_column=c, normalize_column=normalize_column) \
                                .fit(df_train[msk]))

    df_train.drop([c for c in df_train.columns if '__p_click' in c], axis=1, inplace=True)
    df_test.drop([c for c in df_test.columns if '__p_click' in c], axis=1, inplace=True)

    for estimator, column_name in zip(unary_estimators, estimator_columns):
        df_train = df_train.join(
            estimator.predict(df_train, bin_count=bins, bin_type=bin_type).add_prefix(column_name + '__'))
        df_test = df_test.join(
            estimator.predict(df_test, bin_count=bins, bin_type=bin_type).add_prefix(column_name + '__'))

    return df_train, df_test, unary_estimators


def train_lr_and_show_results(df_train, df_test, msk, subject, measure=ProcessMeasure(), C=None):
    msk = msk if msk is not None else np.zeros(df_train.shape[0]).astype(bool)
    X_train = df_train[[c for c in df_train.columns if '__p_click' in c]].to_numpy()
    y_train = df_train.click.to_numpy().astype('int')
    X_test = df_test[[c for c in df_test.columns if '__p_click' in c]].to_numpy()  # ohe.transform(df_test)
    y_test = df_test.click.to_numpy().astype('int')

    measure.start(subject)
    lr = LogisticRegression(random_state=0, max_iter=10000, verbose=0, solver='lbfgs', C=C).fit(X_train[~msk],
                                                                                                y_train[~msk])
    measure.stop(subject)
    auc = results_f(lr, X_train[~msk], y_train[~msk], X_test, y_test, subject)
    measure.data_point(auc, collection='auc_{}'.format(subject))
    measure.print()

    return lr
