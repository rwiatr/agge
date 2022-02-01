import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from experiment.ipinyou import conjunction
from experiment.ipinyou.CONST import CACHE_DIR
from experiment.ipinyou.agge.agge_handle import AggeHandle
from experiment.ipinyou.hash.encoder import HashFeatureEncoder, OHtoHT
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge.run import neg_sample
import scipy.sparse as sparse
import random

from experiment.ipinyou.sets import SetHandler


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

        self.sample_handler = None

    def get_data(self, subject, bins, sample_id, conj=False, agge_conj=True, agge=True, ht=True):
        print(f"subject={subject}, bins={bins}, sample_id={sample_id}")
        self.new_subject = subject != self.prev_subject
        self.new_sample = sample_id != self.prev_sample_id
        self.new_bins = bins != self.prev_bins

        self.prev_subject = subject
        self.prev_bins = bins
        self.prev_sample_id = sample_id
        self.cols = ['weekday', 'hour',  # 'timestamp',
                     'useragent', 'region', 'city', 'adexchange',
                     'slotwidth', 'slotheight',
                     'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                     'creative',  # 'bidprice', #'payprice',
                     'keypage', 'advertiser']
        if self.new_subject:
            self.sample_handler = SetHandler(subject, CACHE_DIR, self.cols, 4250).sample_handler(
                ht=ht,
                conj=conj,
                agge=agge,
                conj_agge=agge_conj,
                sample_id=sample_id)

        if self.new_subject or self.new_sample or self.new_bins:
            sampling = 0.2 if subject in {'1458', '3386'} else 0.5
            self.capture = None
            self.capture = self.sample_handler(bins, sampling, sample_id)
        else:
            print("SKIP ENCODING, USING CAPTURED")

        return self.capture['X'], self.capture['y']

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


def datasets(X, y, agge=False):
    if agge:
        return ({"train": X['train_agge'].toarray(), "test": X['test_agge'].toarray()},
                {"train": y['train'], "test": y['test']})
    return ({"train": X['train_one_hot'], "test": X['test_one_hot']},
            {"train": y['train'], "test": y['test']})


def datasets_onehot_conj(X, y, agge=False):
    if agge:
        return ({"train": X['train_long_agge'].toarray(), "test":  X['test_long_agge'].toarray()},
                {"train": y['train'], "test": y['test']})
    return ({"train": X['train_long_ht'], "test":  X['test_long_ht']},
            {"train": y['train'], "test": y['test']})


def __merge(csr0, csr1):
    return scipy.sparse.hstack([csr0, csr1], format="csr")
