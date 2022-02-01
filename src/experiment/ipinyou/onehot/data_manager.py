import numpy as np
from sklearn.preprocessing import OneHotEncoder
from experiment.ipinyou import conjunction
from experiment.ipinyou.agge.agge_handle import AggeHandle
from experiment.ipinyou.hash.encoder import HashFeatureEncoder, OHtoHT
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge.run import neg_sample
import scipy.sparse as sparse
import random


class DataManager:

    def __init__(self):
        self.prev_subject = None
        self.prev_sample_id = None
        self.prev_bins = None

        self.df_train = None
        self.df_test = None
        self.capture = None

        self._df_test = None
        self._df_train = None

        self.new_subject = False
        self.new_sample = False
        self.new_bins = False

    def get_data(self, subject, bins, sample_id, conj=False, agge_conj=True, fast_ht=False):
        print(f"subject={subject}, bins={bins}, sample_id={sample_id}")
        self.new_subject = subject != self.prev_subject
        self.new_sample = sample_id != self.prev_sample_id
        self.new_bins = bins != self.prev_bins

        self.prev_subject = subject
        self.prev_bins = bins
        self.prev_sample_id = sample_id

        if self.new_subject:
            self.df_train, self.df_test = read_data(subject)
            self.df_train.drop(columns=['usertag'], inplace=True)
            self.df_test.drop(columns=['usertag'], inplace=True)

        if self.new_sample:
            if subject in {'1458', '3386'}:
                self._df_test = neg_sample(self.df_test, 0.2)
                self._df_train = neg_sample(self.df_train, 0.2)
            else:
                self._df_test = neg_sample(self.df_test, 0.5)
                self._df_train = neg_sample(self.df_train, 0.5)

            self.cols = ['weekday', 'hour',  # 'timestamp',
                         'useragent', 'region', 'city', 'adexchange',
                         'slotwidth', 'slotheight',
                         'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                         'creative',  # 'bidprice', #'payprice',
                         'keypage', 'advertiser']

            self._df_train = self._df_train[self.cols + ['click']]
            self._df_test = self._df_test[self.cols + ['click']]

            self.conj_cols = []
            if conj:
                self._df_train, self.cols, self.conj_cols = conjunction.make_conjunction_features(self._df_train)
                self._df_test, self.cols, self.conj_cols = conjunction.make_conjunction_features(self._df_test)
            self.all_cols = self.cols + self.conj_cols

        if self.new_sample or self.new_subject or self.new_bins:
            y_train = self._df_train.click.to_numpy().astype('float64')
            y_test = self._df_test.click.to_numpy().astype('float64')

            print('ENCODING...')
            X_test, X_train, cat = self.one_hot_encode(self.cols)
            if conj:
                X_test_all, X_train_all, cat = self.one_hot_encode(self.all_cols)
                X_test_conj, X_train_conj, cat = self.one_hot_encode(self.conj_cols)
            else:
                X_test_all, X_train_all = sparse.hstack((X_test, X_test)).tocsr(), \
                                          sparse.hstack((X_train, X_train)).tocsr()
                X_test_conj, X_train_conj = X_test, X_train
            print('ENCODED...', X_test.shape[1], X_test_conj.shape[1], X_test_all.shape[1])
            print('HT ENCODING...')

            if fast_ht:
                seed = str(random.randint(-100000, 1000000)) + "_universal_seed"
                X_test_ht, X_train_ht = self.fast_hashing_trick(X_test, X_train, seed, cat)
                if conj:
                    X_test_ht_all, X_train_ht_all = self.fast_hashing_trick(X_test_all, X_train_all, seed, cat)
                    X_test_ht_conj, X_train_ht_conj = self.fast_hashing_trick(X_test_conj, X_train_conj, seed, cat)
                else:
                    X_test_ht_all, X_train_ht_all = sparse.hstack((X_test, X_test)).tocsr(), \
                                                    sparse.hstack((X_train, X_train)).tocsr()
                    X_test_ht_conj, X_train_ht_conj = X_test, X_train
            else:
                X_test_ht, X_train_ht = self.hashing_trick(self.cols)
                if conj:
                    X_test_ht_all, X_train_ht_all = self.hashing_trick(self.all_cols)
                    X_test_ht_conj, X_train_ht_conj = self.hashing_trick(self.conj_cols)
                else:
                    X_test_ht_all, X_train_ht_all = sparse.hstack((X_test, X_test)).tocsr(), \
                                                    sparse.hstack((X_train, X_train)).tocsr()
                    X_test_ht_conj, X_train_ht_conj = X_test, X_train

            print('AGGE ENCODING ...')
            X_test_agge, X_train_agge = self.aggregate_encode(bins, self.cols)
            if conj and agge_conj:
                X_test_agge_all, X_train_agge_all = self.aggregate_encode(bins, self.all_cols)
                X_test_agge_conj, X_train_agge_conj = self.aggregate_encode(bins, self.conj_cols)
            else:
                X_test_agge_all, X_train_agge_all = np.concatenate((X_test_agge, X_test_agge), axis=1), \
                                                    np.concatenate((X_train_agge, X_train_agge), axis=1)
                X_test_agge_conj, X_train_agge_conj = X_test_agge, X_train_agge
            print('ENCODING FINISHED!')

            self.capture = {
                               "train": X_train, "test": X_test,
                               "train_conj": X_train_conj, "test_conj": X_test_conj,
                               "train_all": X_train_all, "test_all": X_test_all,

                               "train_ht": X_train_ht, "test_ht": X_test_ht,
                               "train_ht_conj": X_train_ht_conj, "test_ht_conj": X_test_ht_conj,
                               "train_ht_all": X_train_ht_all, "test_ht_all": X_test_ht_all,

                               "train_agge": X_train_agge, "test_agge": X_test_agge,
                               "train_agge_conj": X_train_agge_conj, "test_agge_conj": X_test_agge_conj,
                               "train_agge_all": X_train_agge_all, "test_agge_all": X_test_agge_all,
                           }, {
                               "train": y_train, "test": y_test,
                           }, self.cols, self.conj_cols
        else:
            print("SKIP ENCODING, USING CAPTURED")

        return self.capture

    def aggregate_encode(self, bins, cols):
        agge_handler = AggeHandle(bins=bins)
        X_train_agge, X_test_agge = \
            agge_handler.fit_and_convert(self._df_train[cols + ['click']], self._df_test[cols + ['click']], cols)
        return X_test_agge, X_train_agge

    def one_hot_encode(self, cols):
        enc = OneHotEncoder(handle_unknown='ignore')
        ohe = enc.fit(self._df_train[cols])
        X_train = ohe.transform(self._df_train[cols])
        X_test = ohe.transform(self._df_test[cols])

        return X_test, X_train, enc.categories_

    def hashing_trick(self, cols):
        ht = HashFeatureEncoder(columns=cols, hash_space=len(cols) * 10)
        ht.fit(self._df_train[cols])
        X_train = ht.transform(self._df_train[cols])
        X_test = ht.transform(self._df_test[cols])

        return X_test, X_train

    def fast_hashing_trick(self, ohe_train, ohe_test, seed, values):
        ht = OHtoHT(hash_space=ohe_train.shape[1] // 10, seed=seed, values=values)
        X_train = ht.transform(ohe_train)
        X_test = ht.transform(ohe_test)

        return X_test, X_train


def datasets(X, y, agge=False, ht_conj=False):
    if agge:
        return ({"train": X['train_agge'], "test": X['test_agge'],
                 "train_conj": X['train_agge_conj'], "test_conj": X['test_agge_conj'],
                 "train_and_conj": X['train_agge_all'], "test_and_conj": X['test_agge_all']},
                {"train": y['train'], "test": y['test']})
    if ht_conj:
        return ({"train": X['train'], "test": X['test'],
                 "train_conj": X['train_ht_conj'], "test_conj": X['train_ht_conj'],
                 "train_and_conj": X['train_ht_all'], "test_and_conj": X['test_ht_all']},
                {"train": y['train'], "test": y['test']})
    return ({"train": X['train'], "test": X['test'],
             "train_conj": X['train_conj'], "test_conj": X['test_conj'],
             "train_and_conj": X['train_all'], "test_and_conj": X['test_all']},
            {"train": y['train'], "test": y['test']})
