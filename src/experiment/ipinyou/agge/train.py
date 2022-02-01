from time import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment.display import show_auc
from estimators import UnaryEstimator
from experiment.measure import ProcessMeasure
import scipy.sparse


def results_f(lr, X_train, y_train, X_test, y_test, subject):
    show_auc(lr, X_train, y_train, name=subject)
    return show_auc(lr, X_test, y_test, name=subject)


def train_estimators(df_train, df_test, msk=None, normalize_column='minmax', bins=None, bin_type='cnt',
                     estimator_columns=None):
    estimator_columns = ['weekday', 'hour',  # 'timestamp',
                         'useragent', 'region', 'city', 'adexchange',
                         'slotwidth', 'slotheight',
                         'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                         'creative',  # 'bidprice', #'payprice',
                         'keypage', 'advertiser'] if estimator_columns is None else estimator_columns

    unary_estimators = []
    for c in estimator_columns:
        unary_estimators.append(UnaryEstimator(value_column='click', key_column=c, normalize_column=normalize_column) \
                                .fit(df_train))

    df_train.drop([c for c in df_train.columns if '__p_click' in c], axis=1, inplace=True)
    df_test.drop([c for c in df_test.columns if '__p_click' in c], axis=1, inplace=True)

    start = time()
    for idx, (estimator, column_name) in enumerate(zip(unary_estimators, estimator_columns)):
        df_train = df_train.join(
            estimator.predict(df_train, bin_count=bins, bin_type=bin_type).add_prefix(column_name + '__'))
        df_test = df_test.join(
            estimator.predict(df_test, bin_count=bins, bin_type=bin_type).add_prefix(column_name + '__'))
        if (idx + 1) % 10 == 0:
            total_time = time() - start
            estimated_time_left = total_time / (idx + 1) * len(estimator_columns)
            print(
                f">>>> predicted {(idx + 1)}/{len(estimator_columns)} columns, {total_time / (idx + 1) :.2f}s/column, "
                f"estimated time left: {estimated_time_left:.2f}s")

    return df_train, df_test, unary_estimators


def train_estimators2(df_train, df_test, msk=None, normalize_column='minmax', bins=None, bin_type='cnt',
                      estimator_columns=None):
    estimator_columns = ['weekday', 'hour',  # 'timestamp',
                         'useragent', 'region', 'city', 'adexchange',
                         'slotwidth', 'slotheight',
                         'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                         'creative',  # 'bidprice', #'payprice',
                         'keypage', 'advertiser'] if estimator_columns is None else estimator_columns

    unary_estimators = []
    for c in estimator_columns:
        unary_estimators.append(UnaryEstimator(value_column='click', key_column=c, normalize_column=normalize_column) \
                                .fit(df_train))

    df_train.drop([c for c in df_train.columns if '__p_click' in c], axis=1, inplace=True)
    df_test.drop([c for c in df_test.columns if '__p_click' in c], axis=1, inplace=True)

    start = time()
    train_cols = []
    test_cols = []
    col_size = 0
    train_cols_csc = None
    test_cols_csc = None
    for idx, (estimator, column_name) in enumerate(zip(unary_estimators, estimator_columns)):
        train_col = estimator.predict2(df_train, bin_count=bins, bin_type=bin_type)
        train_cols.append(train_col.toarray())
        test_col = estimator.predict2(df_test, bin_count=bins, bin_type=bin_type)
        test_cols.append(test_col.toarray())
        col_size += train_col.shape[1]
        # if test_cols_csc is None:
        #     train_cols_csc = train_col
        #     test_cols_csc = test_col
        # else:
        #     train_cols_csc = scipy.sparse.hstack([train_cols_csc, train_col], format="csc", dtype=np.float32)
        #     test_cols_csc = scipy.sparse.hstack([test_cols_csc, test_col], format="csc", dtype=np.float32)
        if (idx + 1) % 10 == 0:
            total_time = time() - start
            estimated_time_left = total_time / (idx + 1) * len(estimator_columns)
            print(f">>>> predicted {(idx + 1)}/{len(estimator_columns)} columns, "
                  f"{total_time / (idx + 1) :.2f}s/column, "
                  f"estimated time left: {estimated_time_left:.2f}s")
    print(f"columns={col_size}")
    # return train_cols_csc, test_cols_csc, unary_estimators
    # return scipy.sparse.hstack([
    #     scipy.sparse.hstack(train_cols[:len(train_cols) // 2], format="csr", dtype=np.float32),
    #     scipy.sparse.hstack(train_cols[len(train_cols) // 2:], format="csr", dtype=np.float32)
    # ], format="csr", dtype=np.float32), scipy.sparse.hstack([
    #     scipy.sparse.hstack(test_cols[:len(test_cols) // 2], format="csr", dtype=np.float32),
    #     scipy.sparse.hstack(test_cols[len(test_cols) // 2:], format="csr", dtype=np.float32)
    # ], format="csr", dtype=np.float32), unary_estimators
    # return scipy.sparse.hstack(train_cols, format="csr", dtype=np.float32), \
    #        scipy.sparse.hstack(test_cols, format="csr", dtype=np.float32), \
    #        unary_estimators
    # return scipy.sparse.csr_matrix(np.hstack(train_cols), dtype=np.float32), \
    #        scipy.sparse.csr_matrix(np.hstack(test_cols), dtype=np.float32), \
    #        unary_estimators
    start = time()
    try:
        return __large_to_csr(np.hstack(train_cols), parts=10), \
               __large_to_csr(np.hstack(test_cols), parts=10), \
               unary_estimators
    finally:
        print(f"translating values in {time() - start:.2f}s")


def train_estimators3(df_train, df_test, set_consumer, parts=10, msk=None, normalize_column='minmax',
                      bins=None, bin_type='cnt',
                      estimator_columns=None):
    estimator_columns = ['weekday', 'hour',  # 'timestamp',
                         'useragent', 'region', 'city', 'adexchange',
                         'slotwidth', 'slotheight',
                         'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                         'creative',  # 'bidprice', #'payprice',
                         'keypage', 'advertiser'] if estimator_columns is None else estimator_columns

    unary_estimators = []
    for c in estimator_columns:
        unary_estimators.append(UnaryEstimator(value_column='click', key_column=c, normalize_column=normalize_column) \
                                .fit(df_train))

    return unary_estimators, estimator_columns


def __large_to_csr(arr, parts=10):
    part_size = (arr.shape[0] // parts)
    start = time()
    stacked = scipy.sparse.vstack(
        [scipy.sparse.csr_matrix(
            arr[part_id * part_size:max((part_id + 1) * part_size, arr.shape[0])]
        ) for part_id in range(parts + 1)],
        format="csr")
    print(f"stacking time {time() - start:.2f}s")
    return stacked


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
