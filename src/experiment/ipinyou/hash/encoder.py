import time
import scipy.sparse
import numpy as np


class HashFeatureEncoder:

    def __init__(self, columns=None, array_columns=None, hash_space=101, alternate_sign=True):
        self.columns = columns if columns is not None else []
        self.array_columns = array_columns if array_columns is not None else []
        self.hash_space = hash_space
        self.alternate_sign = alternate_sign
        self.transformers = None

    def fit(self, df):
        return self

    def transform(self, df):
        print('encode ...')
        single_space = type(self.hash_space) == int
        if single_space:
            result = np.zeros((df.shape[0], self.hash_space))
            hash_space = [self.hash_space for _ in self.columns]
        else:
            result = []
            hash_space = self.hash_space

        total = len(self.columns) + len(self.array_columns)
        for idx, hash_column in enumerate(zip(self.columns, hash_space)):
            column, n_features = hash_column
            from sklearn.feature_extraction import FeatureHasher
            h = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=self.alternate_sign)
            salt = str(hash(column))
            f = h.transform(df[column].astype(str).apply(lambda x: [x + salt]))  # FeatureHasher requires vectors
            if single_space:
                result = result + f.toarray()
            else:
                result.append(f.toarray())

        for idx, hash_column in enumerate(zip(self.array_columns, hash_space)):
            column, n_features = hash_column
            h = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=self.alternate_sign)
            f = h.transform(df[column])
            if single_space:
                result = result + f.toarray()
            else:
                result.append(f.toarray())

        if single_space:
            return result

        # if self.sparse:
        #     return hstack(result)
        return np.concatenate(result, axis=1)

    def get_index(self, column, value):
        return list(self.transformers[column].classes_).index(value)


class OHtoHT:

    def __init__(self, hash_space=101, alternate_sign=True, seed="some_long_string", seed2=None, values=None):
        self.hash_space = hash_space
        self.alternate_sign = alternate_sign
        self.seed = seed
        self.seed2 = seed2 if seed2 != None else seed + "_additional_seed"
        self.values = values

    def fit(self, oh):
        return self

    def transform(self, oh):
        start = time.time()
        print('convert csr ...')
        oh = oh.tocsc()
        print(f'in {time.time() - start}s')
        print('encode fast ht ... ', oh.shape)

        result = np.zeros((oh.shape[0], self.hash_space))
        start = time.time()

        for dim in range(oh.shape[1]):
            idx = abs(hash(self.seed + str(dim) + self.seed)) % self.hash_space
            mod = 1 if not self.alternate_sign else np.sign(hash(self.seed2 + str(dim) + self.seed2))
            mod = 1 if mod == 0 else mod
            # result[:, idx] += oh[:, dim].toarray().reshape(-1) * mod
            result[:, idx] += oh.getcol(dim).toarray().reshape(-1) * mod

            if dim % 5000 == 0:
                print(f'encode fast ht ... {dim}/{oh.shape[1]} in {time.time() - start:.2f}s')
                start = time.time()

        return scipy.sparse.csc_matrix(result)
