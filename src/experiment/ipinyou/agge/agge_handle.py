from experiment.ipinyou.agge.train import train_estimators

class AggeHandle:

    def __init__(self, bins):
        self.bins = bins

    def fit_and_convert(self, df_train, df_test):
        df_train, df_test, _ = train_estimators(df_train, df_test, None,
                                                normalize_column='minmax',
                                                bins=self.bins,
                                                bin_type='qcut')
        return df_train[[c for c in df_train.columns if '__p_click' in c]].to_numpy(), \
               df_test[[c for c in df_test.columns if '__p_click' in c]].to_numpy()