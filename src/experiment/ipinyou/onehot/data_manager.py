from sklearn.preprocessing import OneHotEncoder

from experiment.ipinyou.agge.agge_handle import AggeHandle
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge.run import neg_sample

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


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

    def get_data(self, subject, bins, sample_id):
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

        cols = ['weekday', 'hour',  # 'timestamp',
                'useragent', 'region', 'city', 'adexchange',
                'slotwidth', 'slotheight',
                'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                'creative',  # 'bidprice', #'payprice',
                'keypage', 'advertiser']


        dense_features = ['weekday', 'hour', 'region',  'city', 'slotwidth', 'slotheight', 'slotprice_bucket']
        sparse_features = [col_name for col_name in cols if col_name not in dense_features]

        if self.new_sample or self.new_subject or self.new_bins:
            print('ENCODING...')
            enc = OneHotEncoder(handle_unknown='ignore')
            ohe = enc.fit(self._df_train[cols])

            X_train = ohe.transform(self._df_train[cols])
            y_train = self._df_train.click.to_numpy().astype('float64')

            X_test = ohe.transform(self._df_test[cols])
            y_test = self._df_test.click.to_numpy().astype('float64')

            print('AGGE ENCODING ...')

            agge_handler = AggeHandle(bins=bins)
            X_train_agge, X_test_agge = \
                agge_handler.fit_and_convert(self._df_train[cols + ['click']], self._df_test[cols + ['click']])


            print('LABEL ENCODING AND TRANSFORMATION')
            linear_features_columns_list, dnn_features_columns_list, model_inputs = self.get_sparse_dense_data(dense_features, sparse_features)


            print('ENCODING FINISHED!')

            self.capture = X_train, y_train, X_test, y_test, X_train_agge, X_test_agge, linear_features_columns_list, dnn_features_columns_list, model_inputs
        else:
            print("SKIP ENCODING, USING CAPTURED")
        
        return self.capture

    def get_sparse_dense_data(self, dense_features, sparse_features):

        dnn_feature_columns_list = []
        linear_feature_columns_list = []
        model_inputs = []

        for data in [self._df_train, self._df_test]:
            # 1.Label Encoding for sparse features,and do simple Transformation for dense features
            for feat in sparse_features:
                lbe = LabelEncoder()
                data[feat] = lbe.fit_transform(data[feat])
            mms = MinMaxScaler(feature_range=(0, 1))
            data[dense_features] = mms.fit_transform(data[dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name

            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                    for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                    for feat in dense_features]

            dnn_feature_columns_list += [fixlen_feature_columns]
            linear_feature_columns_list += [fixlen_feature_columns]

            feature_names = get_feature_names(
                fixlen_feature_columns + fixlen_feature_columns)

            model_inputs += [{name: data[name] for name in feature_names}]

        return linear_feature_columns_list, dnn_feature_columns_list, model_inputs