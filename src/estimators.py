import pandas as pd
import numpy as np
import time

import scipy.sparse


def cnt_bins(p, attempts, bins=3, type='log', echo=False):
    if type == 'log':
        return np.log10(attempts).astype(int)
    if type == 'div':
        return (attempts / 50).astype(int)
    if type == 'qcut':
        # return pd.cut(attempts, bins=bins, labels=[x for x in range(bins)], precision=8).astype(float)
        bins = min(bins, attempts.unique().shape[0])
        bins = pd.qcut(attempts, bins, duplicates='drop', retbins=False)
        if echo:
            print('unique values {}, bins {} '.format(attempts.unique().shape[0], bins.shape[0]))
        unique = np.unique(bins)
        result = np.zeros(bins.shape)

        for bin_id, a_bin in enumerate(unique):
            result[bins == a_bin] = bin_id

        return pd.Series(result.astype(float))

    if type == 'cnt':
        # return pd.cut(attempts, bins=bins, labels=[x for x in range(bins)], precision=8).astype(float)
        bins = min(bins, attempts.unique().shape[0])
        return pd.cut(attempts, bins=bins, labels=[x for x in range(bins)]).astype(float)


def confidence_bins(p, attempts, bins=3, confidence_level=0.80):
    left_ok = p * attempts > 5  # successes
    right_ok = (1 - p) * attempts > 5  # failures

    ok = left_ok | right_ok

    z = 1 - (1 - confidence_level) / 2
    confidence_interval = z * np.sqrt(p * (1 - p) / attempts)

    confidence_bean = pd.cut(confidence_interval, bins=bins, labels=[x for x in range(bins)]).astype(float)

    zero = p == 0
    one = p == 1
    ok = ok & (zero == False) & (one == False)

    zero_ok = zero & attempts > 30
    one_ok = one & attempts > 30
    ok = zero_ok | one_ok | ok
    not_ok = ok == False
    # print('OKs {}; nOKs {}; 0s {}; 1s {}'.format(ok.sum(), not_ok.sum(), zero.sum(), one.sum()))

    confidence_bean[not_ok] = len(confidence_bean.unique())
    return confidence_bean


def f_bins(p, attempts, bins=3, bin_type='cnt'):
    # result = cnt_beans_2(p, attempts, bins=bins)
    if bin_type in ['cnt', 'log', 'div', 'qcut']:
        result = cnt_bins(p, attempts, bins=bins, type=bin_type)
    elif bin_type == 'confidence':
        result = confidence_bins(p, attempts, bins=bins)
    else:
        raise Exception("no such method ({})".format(bin_type))
    if result.unique().shape == 1:
        result[:] = 0
    else:
        for idx, b in enumerate(result.unique()):
            result[result == b] = idx
    # print('unique bins ' + str(len(result.unique())))
    # print('count per bin ' + str(result.value_counts()))
    return result


class MultiColEstimator:
    def __init__(self, value_column, key_columns, normalize_column):
        self.value_column = value_column
        self.key_columns = key_columns
        self.normalize_column = normalize_column
        self.model = None

    def fit(self, df):
        self.model = pd.DataFrame({self.value_column: df[self.value_column].astype('float64'),
                                   self.key_column: df[self.key_column]})
        self.model['estimator_event_col'] = 1
        self.model = self.model.groupby(self.key_column).sum().reset_index()
        return self

    def normalize(self, estimation, column):
        if self.normalize_column == 'meanstd':
            if estimation[column].std() == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = (estimation[column] - estimation[column].mean()) / estimation[
                    column].std()
        if self.normalize_column == 'minmax':
            if (estimation[column].max() - estimation[column].min()) == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = (estimation[column] - estimation[column].min()) / (
                        estimation[column].max() - estimation[column].min())
        if self.normalize_column == 'zeromax':
            if estimation[column].max() == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = estimation[column] / estimation[column].max()
        estimation[column].fillna(0, inplace=True)

    def predict(self, df, bin_count=None, bin_type='cnt'):
        result_columns = []
        estimation = df[[self.key_column]]

        if bin_count is not None and bin_count > 0:
            self.model['bin'] = f_bins(self.model[self.value_column] / self.model['estimator_event_col'],
                                       self.model['estimator_event_col'], bins=bin_count, bin_type=bin_type)
        else:
            self.model['bin'] = 0

        to_drop = [column for column in self.model.columns if column is not self.key_column]
        estimation = estimation.drop(to_drop, errors='ignore', axis=1)
        estimation = estimation.merge(self.model, on=self.key_column, how='left')

        pcolumn = 'p_' + self.value_column
        estimation[pcolumn] = estimation[self.value_column] / estimation['estimator_event_col']
        self.normalize(estimation, pcolumn)
        max_bin = self.model['bin'].max()
        for bin_id in range(int(max_bin + 1)):
            bin_column = pcolumn + '_bin=' + str(bin_id)
            estimation[bin_column] = np.NAN
            estimation.loc[estimation['bin'] == bin_id, bin_column] \
                = estimation.loc[estimation['bin'] == bin_id, pcolumn]
            self.normalize(estimation, bin_column)
            result_columns.append(bin_column)

        return estimation[result_columns]


class UnaryEstimator:
    def __init__(self, value_column, key_column, normalize_column):
        self.value_column = value_column
        self.key_column = key_column
        self.normalize_column = normalize_column
        self.model = None

    def fit(self, df):
        sec, _ = self.__timed(lambda: self.__fit(df))
        # print(f"fitting {self.key_column} took {sec:.2f}s")
        return self

    def __fit(self, df):
        self.model = pd.DataFrame({self.value_column: df[self.value_column].astype('float32'),
                                   self.key_column: df[self.key_column]})
        self.model['estimator_event_col'] = 1
        self.model = self.model.groupby(self.key_column).sum().reset_index()
        return self

    def normalize(self, estimation, column):
        sec, _ = self.__timed(lambda: self.__normalize(estimation, column))
        return sec

    def __normalize(self, estimation, column):
        if self.normalize_column == 'meanstd':
            if estimation[column].std() == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = (estimation[column] - estimation[column].mean()) / estimation[
                    column].std()
        if self.normalize_column == 'minmax':
            if (estimation[column].max() - estimation[column].min()) == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = ((estimation[column] - estimation[column].min()) / (
                        estimation[column].max() - estimation[column].min())).astype('float32')
        if self.normalize_column == 'zeromax':
            if estimation[column].max() == 0:
                estimation.loc[estimation[column].notna(), column] = 1
            else:
                estimation[column] = estimation[column] / estimation[column].max()
        estimation[column].fillna(0, inplace=True)

    def normalize_np(self, estimation, column):
        sec, _ = self.__timed(lambda: self.__normalize_np(estimation, column))
        return sec

    def __normalize_np(self, estimation, column):
        if self.normalize_column == 'minmax':
            column__min = estimation[:, column].min()
            column__max = estimation[:, column].max()

            if (column__max - column__min) == 0:
                estimation[:] = 1
            else:
                estimation[:, column] = \
                    ((estimation[:, column] - column__min) / (column__max - column__min)).astype('float32')

    def predict(self, df, bin_count=None, bin_type='cnt'):
        sec, res = self.__timed(lambda: self.__predict(df, bin_count, bin_type))
        # print(f"predicting {self.key_column} took {sec:.2f}s")
        return res

    def predict2(self, df, bin_count=None, bin_type='cnt'):
        sec, res = self.__timed(lambda: self.__predict2(df, bin_count, bin_type))
        # print(f"predicting {self.key_column} took {sec:.2f}s")
        return res

    def __predict(self, df, bin_count=None, bin_type='cnt'):
        result_columns = []
        # print(f'calculating {self.key_column}')
        estimation = df[[self.key_column]]

        sec, _ = self.__timed(lambda: self.calculate_bins(bin_count, bin_type))
        # print(f"calculating bins for {self.key_column} took {sec:.2f}s")

        to_drop = [column for column in self.model.columns if column is not self.key_column]
        estimation = estimation.drop(to_drop, errors='ignore', axis=1)
        estimation = estimation.merge(self.model, on=self.key_column, how='left')

        pcolumn = 'p_' + self.value_column
        estimation[pcolumn] = estimation[self.value_column] / estimation['estimator_event_col']
        sec = self.normalize(estimation, pcolumn)
        # print(f"normalizing {pcolumn} took {sec :.2f}s")
        max_bin = self.model['bin'].max()
        sec = 0

        # print(f'allocating ({estimation.shape[0], max_bin + 1})')
        # estim = np.ndarray((estimation.shape[0], max_bin + 1), dtype=np.float32)
        # estim[:] = np.NAN
        for bin_id in range(int(max_bin + 1)):
            bin_column = pcolumn + '_bin=' + str(bin_id)
            estimation[bin_column] = np.NAN  # fragmented
            estimation.loc[estimation['bin'] == bin_id, bin_column] \
                = estimation.loc[estimation['bin'] == bin_id, pcolumn]
            sec += self.normalize(estimation, bin_column)
            result_columns.append(bin_column)
        # print(f"normalizing bins {self.key_column} took {sec:.2f}s")

        return estimation[result_columns]

    def __predict2(self, df, bin_count=None, bin_type='cnt'):
        result_columns = []
        estimation = df[[self.key_column]]

        sec, _ = self.__timed(lambda: self.calculate_bins(bin_count, bin_type))
        # print(f"calculating bins for {self.key_column} took {sec:.2f}s")

        estimation = estimation.merge(self.model, on=self.key_column, how='left')

        # TODO do we need this?
        # estimation[pcolumn] = estimation[self.value_column] / estimation['estimator_event_col']
        # sec = self.normalize(estimation, pcolumn)
        # print(f"normalizing {pcolumn} took {sec :.2f}s")
        max_bin = self.model['bin'].max()
        sec = 0
        pcolumn_np = (estimation[self.value_column] / estimation['estimator_event_col']).to_numpy(dtype=np.float32)

        bins_np = estimation['bin'].to_numpy()
        # print(f'allocating ({estimation.shape[0], max_bin + 1})')
        estim = np.zeros((estimation.shape[0], int(max_bin + 1)), dtype=np.float32)
        for bin_id in range(int(max_bin + 1)):
            estim[bins_np == bin_id, bin_id] = pcolumn_np[bins_np == bin_id]
            sec += self.normalize_np(estim, bin_id)
        # print(f"normalizing bins {self.key_column} took {sec:.2f}s")

        return scipy.sparse.csc_matrix(estim, dtype=np.float32)

    def calculate_bins(self, bin_count, bin_type):
        if bin_count is not None and bin_count > 0:
            self.model['bin'] = f_bins(self.model[self.value_column] / self.model['estimator_event_col'],
                                       self.model['estimator_event_col'], bins=bin_count, bin_type=bin_type)
        else:
            self.model['bin'] = 0

    def __timed(self, fn):
        start = time.time()
        result = fn()
        return time.time() - start, result

# class FastAggregate:
#
#     def __init__(self, key):
#         self.key = key
#
#     def fit(self, df):
#         self.agg = df[[self.key, 'success', 'attempt']].group_by(self.key).sum().reset_index()
#         self.agg['p'] = self.agg['success'] / self.agg['attempt']
#         self.agg['p'].fillna(0, inplace=True)
#         if (self.agg['p'].max() - self.agg['p'].min()) == 0:
#             self.agg.loc[self.agg['p'].notna(), 'p'] = 1
#         else:
#             _max = self.agg['p'].max()
#             _min = self.agg['p'].min()
#             self.agg['p'] = (self.agg['p'] - _min) / (_max - _min)
#         self.agg['p'].fillna(0, inplace=True)
#
#     def fit_bins(self, bin_count):
#         self.agg['bin'] = f_bins(None, self.agg['attempt'], bins=bin_count, bin_type='cqut')
#
#     def predict(self, bin_count=None, bin_type='cnt'):
