import time

import numpy as np
import pandas as pd
import scipy.sparse

from experiment.ipinyou.agge.train import train_estimators, train_estimators2, train_estimators3


class AggeHandle:

    def __init__(self, bins):
        self.bins = bins

    def fit_and_convert(self, df_train, df_test, cols):
        df_train, df_test, _ = train_estimators2(df_train, df_test, None,
                                                 normalize_column='minmax',
                                                 bins=self.bins,
                                                 bin_type='qcut',
                                                 estimator_columns=cols)
        if type(df_test) == pd.DataFrame:
            return df_train[[c for c in df_train.columns if '__p_click' in c]].to_numpy(), \
                   df_test[[c for c in df_test.columns if '__p_click' in c]].to_numpy()
        return df_train, df_test

    def fit(self, df_train, df_test, cols):
        return train_estimators3(df_train, df_test, None,
                                 normalize_column='minmax',
                                 bins=self.bins,
                                 bin_type='qcut',
                                 estimator_columns=cols)

    def convert(self, df, estimators, estimator_columns, bin_type='qcut'):
        dat = []
        col_size = 0
        start = time.time()
        for idx, (estimator, column_name) in enumerate(zip(estimators, estimator_columns)):
            d = estimator.predict2(df, bin_count=self.bins, bin_type=bin_type)
            dat.append(d.toarray())
            col_size += d.shape[1]

            if (idx + 1) % 10 == 0:
                total_time = time.time() - start
                estimated_total_time = total_time / (idx + 1) * len(estimator_columns)
                print(f">>>> predicted {(idx + 1)}/{len(estimator_columns)} columns, "
                      f"{total_time / (idx + 1) :.2f}s/column, "
                      f"estimated total time: {estimated_total_time:.2f}s")
        print(f"columns={col_size}")

        return scipy.sparse.csr_matrix(np.hstack(dat), dtype=np.float32)
