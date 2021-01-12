import pandas as pd
import numpy as np

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

class UnaryEstimator:
    def __init__(self, value_column, key_column, normalize_column):
        self.value_column = value_column
        self.key_column = key_column
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