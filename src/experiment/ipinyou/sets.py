import random
import time

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from experiment.ipinyou.agge.agge_handle import AggeHandle
from experiment.ipinyou.conjunction import make_conjunction_features
from experiment.ipinyou.hash.encoder import OHtoHT
from experiment.ipinyou.load import read_data
import os
import scipy.sparse
import numpy as np
import hashlib


class SetHandler:

    def __init__(self, subject, cache_dir, read_columns, hashing_space, reuse_aggregates=True, max_samples=30):
        self.subject = subject
        self.cache_dir = cache_dir
        self.read_columns = [c for c in read_columns if c != 'click']
        self.hashing_space = hashing_space
        self.reuse_aggregates = reuse_aggregates
        self.max_samples = max_samples

    def sample_handler(self, ht=False, conj=True, agge=True, conj_agge=False, sample_id=0):
        # df_train, df_test, X_train_csr_cols, X_test_csr_cols, X_train_ht_csc_conj, \
        # X_test_ht_csc_conj, cols, cols_conj = self.__read(conj=conj, ht=ht)
        df_train, df_test, X_train_csr_cols, X_test_csr_cols, X_train_csr_conj, \
        X_test_csr_conj, cols, cols_conj = self.__read2(conj=conj, ht=ht)

        def sample(bins, neg_sampling, sample_id):
            train_samples, test_samples = self.__get_samples(df_train, df_test, neg_sampling, sample_id)

            if ht:
                seed_a = random.randint(0, 10000)
                seed_b = random.randint(0, 10000)
                X_train_ht_csc_conj = self.__hashing_trick2(X_train_csr_conj, "train.conj", seed_a, seed_b, sample_id) \
                    if conj and ht else None
                X_test_ht_csc_conj = self.__hashing_trick2(X_test_csr_conj, "test.conj", seed_a, seed_b, sample_id) \
                    if conj and ht else None
            else:
                X_train_ht_csc_conj, X_test_ht_csc_conj = None, None

            if agge:
                X_train_agge, X_test_agge, counts_agge = self.__read_and_cache_aggregates(
                    df_train.iloc[train_samples],
                    df_test.iloc[test_samples],
                    bins,
                    cols,
                    sample_id,
                    "agge")
            else:
                X_train_agge, X_test_agge, counts_agge = None, None, None

            if conj and conj_agge and agge:
                X_train_agge_conj, X_test_agge_conj, counts_agge_conj = self.__read_and_cache_aggregates(
                    df_train.iloc[train_samples],
                    df_test.iloc[test_samples],
                    bins,
                    cols_conj,
                    sample_id,
                    "conj")
            else:
                X_train_agge_conj, X_test_agge_conj, counts_agge_conj = None, None, None

            result = {
                "X": {
                    "train_one_hot": X_train_csr_cols[train_samples],
                    "test_one_hot": X_test_csr_cols[test_samples],

                    "train_ht_conj": X_train_ht_csc_conj[train_samples] if conj and ht else None,
                    "test_ht_conj": X_test_ht_csc_conj[test_samples] if conj and ht else None,

                    "train_agge": X_train_agge,
                    "test_agge": X_test_agge,

                    "train_agge_conj": X_train_agge_conj,
                    "test_agge_conj": X_test_agge_conj,

                    "counts_agge_conj": counts_agge_conj,
                    "counts_agge_conj_long":
                        np.concatenate(
                            (np.sum(X_train_csr_cols[train_samples].toarray(), axis=0).reshape(-1), counts_agge_conj),
                            axis=0
                        ),
                    "counts_agge": counts_agge,

                    "train_long_ht": self.__encode_long_onehot_and_additional(X_train_csr_cols, X_train_ht_csc_conj,
                                                                              "hashing", "train", None,
                                                                              self.hashing_space)[train_samples],
                    "test_long_ht": self.__encode_long_onehot_and_additional(X_test_csr_cols, X_test_ht_csc_conj,
                                                                             "hashing", "test", None,
                                                                             self.hashing_space)[test_samples],

                    "train_long_agge": self.__encode_long_onehot_and_additional(X_train_csr_cols[train_samples],
                                                                                X_train_agge_conj,
                                                                                "aggregate", "train", sample_id, bins),
                    "test_long_agge": self.__encode_long_onehot_and_additional(X_test_csr_cols[test_samples],
                                                                               X_test_agge_conj,
                                                                               "aggregate", "test", sample_id, bins)
                },
                "y": {
                    "train": df_train.iloc[train_samples].click.to_numpy(dtype=int),
                    "test": df_test.iloc[test_samples].click.to_numpy(dtype=int)
                }
            }
            print("$$$$ shapes $$$$")
            for key in result['X'].keys():
                if result["X"][key] is not None:
                    print(f'$$$$ {key}={result["X"][key].shape} $$$$')
            return result

        return sample

    def __read_and_cache_aggregates(self, df_train, df_test, bins, columns, sample_id, set_type):
        if not self.reuse_aggregates or sample_id is None:
            X_train_agge, X_test_agge = self.__aggregate_encode(df_train,
                                                                df_test,
                                                                bins,
                                                                columns)
        if self.reuse_aggregates and sample_id is not None:
            print(f"reusing aggregates {set_type}")
            X_train_agge, X_test_agge, counts = \
                self.__cached_aggregate_encode2(
                    df_train,
                    df_test,
                    bins,
                    columns,
                    sample_id,
                    set_type)
        print(f"aggregate features: {X_train_agge.shape[1]}")
        return X_train_agge, X_test_agge, counts

    def __read(self, conj=True, ht=True):
        df_train, df_test = self.__read_data()
        y_train = df_train.click
        y_test = df_test.click

        df_train = df_train[self.read_columns]
        df_test = df_test[self.read_columns]

        df_train = self.__get_conjunctions(df_train, "train") if conj else df_train
        df_test = self.__get_conjunctions(df_test, "test") if conj else df_test

        cols, cols_conj = [c for c in df_train.columns if "__conj" not in c], \
                          [c for c in df_train.columns if "__conj" in c]

        X_train_csr_cols, X_test_csr_cols = self.__one_hot(df_train[cols], df_test[cols], "cols")
        if conj:
            X_train_csr_conj, X_test_csr_conj = self.__one_hot(df_train[cols_conj], df_test[cols_conj], "conj")
        else:
            X_train_csr_conj, X_test_csr_conj = None, None

        seed_a = random.randint(0, 10000)
        seed_b = random.randint(0, 10000)

        # _ = self.__hashing_trick(X_train_csr_cols, "cols.conj", seed_a, seed_b) if conj else None
        X_train_ht_csc_conj = self.__hashing_trick(X_train_csr_conj, "train.conj", seed_a, seed_b) \
            if conj and ht else None
        X_test_ht_csc_conj = self.__hashing_trick(X_test_csr_conj, "test.conj", seed_a, seed_b) \
            if conj and ht else None

        df_train['click'] = y_train
        df_test['click'] = y_test

        return df_train, df_test, X_train_csr_cols, X_test_csr_cols, \
               X_train_ht_csc_conj, X_test_ht_csc_conj, cols, cols_conj

    def __read2(self, conj=True, ht=True):
        df_train, df_test = self.__read_data()
        y_train = df_train.click
        y_test = df_test.click

        df_train = df_train[self.read_columns]
        df_test = df_test[self.read_columns]

        df_train = self.__get_conjunctions(df_train, "train") if conj else df_train
        df_test = self.__get_conjunctions(df_test, "test") if conj else df_test

        cols, cols_conj = [c for c in df_train.columns if "__conj" not in c], \
                          [c for c in df_train.columns if "__conj" in c]

        X_train_csr_cols, X_test_csr_cols = self.__one_hot(df_train[cols], df_test[cols], "cols")
        if conj:
            X_train_csr_conj, X_test_csr_conj = self.__one_hot(df_train[cols_conj], df_test[cols_conj], "conj")
        else:
            X_train_csr_conj, X_test_csr_conj = None, None

        seed_a = random.randint(0, 10000)
        seed_b = random.randint(0, 10000)

        df_train['click'] = y_train
        df_test['click'] = y_test

        return df_train, df_test, X_train_csr_cols, X_test_csr_cols, \
               X_train_csr_conj, X_test_csr_conj, cols, cols_conj

    def __read_data(self):
        return self.__read_file2(f"{self.cache_dir}/raw.{self.subject}.train.pd.pickle",
                                 f"{self.cache_dir}/raw.{self.subject}.test.pd.pickle",
                                 calc_fn=lambda: read_data(self.subject))

    def __hashing_trick(self, csr, set_name, seed_a, seed_b):
        max_rows = 100000
        transformer = OHtoHT(hash_space=self.hashing_space,
                             seed=f"local_seed{seed_a}local_seed",
                             seed2=f"additional{seed_b}additional")

        csc_list = []
        parts = (csr.shape[0] // max_rows) + 1
        for part in range(parts):
            start_idx = part * max_rows
            end_idx = min((part + 1) * max_rows, csr.shape[0])
            csc_list.append(
                self.__read_file(f"{self.cache_dir}/ht.{self.subject}.{set_name}.{self.hashing_space}"
                                 f".part_{part + 1}_of_{parts}.npz",
                                 calc_fn=lambda: transformer.transform(csr[start_idx:end_idx])))

        return scipy.sparse.vstack(csc_list).tocsc()

    def __hashing_trick2(self, csr, set_name, seed_a, seed_b, sample_id):
        max_rows = 100000
        transformer = OHtoHT(hash_space=self.hashing_space,
                             seed=f"local_seed{seed_a}local_seed",
                             seed2=f"additional{seed_b}additional")

        def __read_parts():
            csc_list = []
            parts = (csr.shape[0] // max_rows) + 1
            for part in range(parts):
                start_idx = part * max_rows
                end_idx = min((part + 1) * max_rows, csr.shape[0])
                csc_list.append(
                    self.__read_file(self.__path(path=f"hashingtrick/{set_name}/sampleid={sample_id}",
                                                 name=f'train.part_{part + 1}_of_{parts}',
                                                 extension="npz"),
                                     calc_fn=lambda: transformer.transform(csr[start_idx:end_idx])))
            return scipy.sparse.vstack(csc_list).tocsc()

        return self.__read_file(self.__path(path=f"hashingtrick/{set_name}/sampleid={sample_id}",
                                            name=f'train.full',
                                            extension="npz"),
                                calc_fn=lambda: __read_parts())

    def __one_hot(self, df_train, df_test, set_name):
        def fit_and_calc():
            enc = OneHotEncoder(handle_unknown='ignore').fit(df_train)
            return enc.transform(df_train).astype('float32'), enc.transform(df_test).astype('float32')

        # file_a, file_b = self.__read_file2(f"{self.cache_dir}/oh.{self.subject}.{set_name}.train.npz",
        #                                    f"{self.cache_dir}/oh.{self.subject}.{set_name}.test.npz",
        #                                    calc_fn=fit_and_calc)

        file_a, file_b = self.__read_file2(
            self.__path(path=f"onehot/{set_name}", name=f'train', extension="npz"),
            self.__path(path=f"onehot/{set_name}", name=f'test', extension="npz"),
            calc_fn=fit_and_calc)

        return file_a.astype('float32'), file_b.astype('float32')

    def __get_conjunctions(self, df, set_type):
        def make_conj():
            result, _, _ = make_conjunction_features(df)
            return result

        file_name = f"{self.cache_dir}/conjunction_df.{self.subject}.{set_type}.pd.pickle"
        data = self.__read_file(file_name, calc_fn=make_conj)
        return data

    def __cached_aggregate_encode(self, df_train, df_test, bins, cols, sample_id, agge_type):
        col_hash = abs(sum([int(hashlib.sha1(col.encode("utf-8")).hexdigest(), 16) for col in cols]) % 10000)
        print(f"{col_hash} from {cols}")
        train, test = self.__read_file2(
            f"{self.cache_dir}/agge.{self.subject}.aggregates_conj"
            f".bins_{bins}.sampleid_{sample_id}.colhash_{col_hash}.{agge_type}.train"
            # f".npy",
            f".npz",
            f"{self.cache_dir}/agge.{self.subject}.aggregates_conj"
            f".bins_{bins}.sampleid_{sample_id}.colhash_{col_hash}.{agge_type}.test"
            f".npz",
            # f".npy",
            lambda: self.__aggregate_encode(df_train,
                                            df_test,
                                            bins,
                                            cols)
        )

        return train, test

    def __cached_aggregate_encode2(self, df_train, df_test, bins, cols, sample_id, agge_type):
        col_hash = abs(sum([int(hashlib.sha1(col.encode("utf-8")).hexdigest(), 16) for col in cols]) % 10000)
        print(f"{col_hash} from {cols}")
        agge_handler = AggeHandle(bins=bins)
        estimators, estimator_columns = agge_handler.fit(df_train, df_test, cols)
        max_rows = 100000

        def __encode_df(df, set_type):
            def __read_parts():
                parts = df.shape[0] // max_rows + 1
                csrs = []
                for part in range(parts):
                    start_idx = part * max_rows
                    end_idx = min((part + 1) * max_rows, df.shape[0])
                    csrs.append(
                        self.__read_file(
                            self.__path(
                                path=f"aggregates/{agge_type}/bins={bins}/colhash={col_hash}/sample={sample_id}",
                                name=f'{set_type}.part_{part + 1}_of_{parts}',
                                extension="npz"),
                            calc_fn=lambda: agge_handler.convert(
                                df[start_idx:end_idx],
                                estimators,
                                estimator_columns
                            )))
                return scipy.sparse.vstack(csrs, format="csr")

            # sec, res = self.__timed(lambda: __read_parts())
            # print(f"read {set_type} parts in {sec:.2f}")
            # return res
            return self.__read_file(
                self.__path(path=f"aggregates/{agge_type}/bins={bins}/colhash={col_hash}/sample={sample_id}",
                            name=f'{set_type}.full',
                            extension="npz"),
                calc_fn=lambda: __read_parts())

        def __encode_counts():
            def __gen():
                for estimator in estimators:
                    estimator.calculate_bins(bin_type='qcut', bin_count=bins)
                    yield estimator.col_counts

            return np.concatenate(list(__gen()), axis=0)

        # D: / proj / phd / cache / 2821 / aggregates / conj / bins = 45 / colhash = 2761 / sample = 0 / train.full.npz, read in 71.28
        # D: / proj / phd / cache / 2821 / aggregates / conj / bins = 45 / colhash = 2761 / sample = 0 / test.full.npz, read in 21.62
        return __encode_df(df_train, set_type="train"), __encode_df(df_test, set_type="test"), __encode_counts()

    def __encode_long_onehot_and_additional(self, oh, second, encoding, set_type, sample_id, bins):
        def __encode_parts():
            max_rows = 100000
            parts = second.shape[0] // max_rows + 1
            csrs = []
            for part in range(parts):
                def __encode_part():
                    start_idx = part * max_rows
                    end_idx = min((part + 1) * max_rows, second.shape[0])
                    return scipy.sparse.hstack([oh[start_idx:end_idx], second[start_idx:end_idx]], format="csr")

                csrs.append(self.__read_file(
                    self.__path(path=f"long_onehot_{encoding}/bins={bins}/sample_id={sample_id}",
                                name=f'{set_type}.part_{part + 1}_of_{parts}',
                                extension="npz"),
                    calc_fn=lambda: __encode_part()))
            return scipy.sparse.vstack(csrs, format="csr")

        return self.__read_file(
            self.__path(path=f"long_onehot_{encoding}/bins={bins}/sample_id={sample_id}",
                        name=f'{set_type}.full',
                        extension="npz"),
            calc_fn=lambda: __encode_parts())

    def __aggregate_encode(self, df_train, df_test, bins, cols):
        agge_handler = AggeHandle(bins=bins)
        X_train_agge, X_test_agge = \
            agge_handler.fit_and_convert(df_train[cols + ['click']], df_test[cols + ['click']], cols)
        return X_train_agge, X_test_agge

    def __timed(self, fn):
        start = time.time()
        result = fn()
        return float(time.time() - start), result

    def __read_cached(self, file_name):
        if file_name.endswith("npz"):
            return scipy.sparse.load_npz(file_name)
        elif file_name.endswith("pd.pickle"):
            return pd.read_pickle(file_name)
        elif file_name.endswith("npy"):
            return np.load(file_name, allow_pickle=True)

    def __write_cached(self, file_name, data):
        idx = file_name.rfind('/')
        path = file_name[:idx]
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        print(f"write type {type(data)} as {file_name}")
        if file_name.endswith("npz"):
            scipy.sparse.save_npz(file_name + ".tmp", data)
            os.rename(file_name + ".tmp.npz", file_name)
        elif file_name.endswith("pd.pickle"):
            data.to_pickle(file_name + ".tmp")
            os.rename(file_name + ".tmp", file_name)
        elif file_name.endswith("npy"):
            np.save(file_name + ".tmp", data)
            os.rename(file_name + ".tmp.npy", file_name)

    def __read_file(self, file_name, calc_fn):
        if os.path.isfile(file_name):
            sec, data = self.__timed(lambda: self.__read_cached(file_name))
            print(f'using cached {file_name}, read in {sec:.2f}s - shape is {data.shape}')
            return data

        sec, data = self.__timed(calc_fn)
        print(f'calculated {file_name} in {sec}s - shape is {data.shape}')

        sec, _ = self.__timed(lambda: self.__write_cached(file_name, data))
        print(f'write {file_name} in {sec}s')

        return data

    def __read_file2(self, file_name_a, file_name_b, calc_fn):
        if os.path.isfile(file_name_a) and os.path.isfile(file_name_b):
            sec_a, data_a = self.__timed(lambda: self.__read_cached(file_name_a))
            sec_b, data_b = self.__timed(lambda: self.__read_cached(file_name_b))
            print(f'using cached {file_name_a} and {file_name_b}, read in {sec_a + sec_b:.2f}s - '
                  f'shapes are {data_a.shape} and {data_b.shape}')
            return data_a, data_b

        sec, (data_a, data_b) = self.__timed(calc_fn)
        print(f'calculated {file_name_a} and {file_name_b} in {sec:.2f}s - '
              f'shapes are {data_a.shape} and {data_b.shape}')

        sec_a, _ = self.__timed(lambda: self.__write_cached(file_name_a, data_a))
        sec_b, _ = self.__timed(lambda: self.__write_cached(file_name_b, data_b))
        print(f'write {file_name_a} and {file_name_b} in {sec_a + sec_b:.2f}s')

        return data_a, data_b

    def __calc_samples(self, df_train, df_test, neg_sampling):
        train_samples = (np.random.rand(df_train.shape[0]) < neg_sampling) | df_train['click'] \
            .to_numpy().astype("bool")
        test_samples = (np.random.rand(df_test.shape[0]) < neg_sampling) | df_test['click'] \
            .to_numpy().astype("bool")

        train_samples = np.arange(0, train_samples.shape[0])[train_samples]
        test_samples = np.arange(0, test_samples.shape[0])[test_samples]

        return train_samples, test_samples

    def __get_samples(self, df_train, df_test, neg_sampling, sample_id):
        if sample_id is None:
            train_samples = (np.random.rand(df_train.shape[0]) < neg_sampling) | df_train['click'] \
                .to_numpy().astype("bool")
            test_samples = (np.random.rand(df_test.shape[0]) < neg_sampling) | df_test['click'] \
                .to_numpy().astype("bool")

            train_samples = np.arange(0, train_samples.shape[0])[train_samples]
            test_samples = np.arange(0, test_samples.shape[0])[test_samples]
        else:
            if sample_id >= self.max_samples:
                raise Exception(f"sample_id={sample_id} >= self.max_samples={self.max_samples}")
            train_samples, test_samples = \
                self.__read_file2(f"{self.cache_dir}/samples.{self.subject}.{sample_id}.{neg_sampling}.train.npy",
                                  f"{self.cache_dir}/samples.{self.subject}.{sample_id}.{neg_sampling}.test.npy",
                                  calc_fn=lambda: self.__calc_samples(df_train, df_test, neg_sampling))

        return train_samples, test_samples

    def __path(self, name, path, extension):
        return f'{self.cache_dir}/{self.subject}/{path}/{name}.{extension}'
