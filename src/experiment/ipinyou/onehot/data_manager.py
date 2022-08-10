from sklearn.preprocessing import OneHotEncoder

from experiment.ipinyou.agge.agge_handle import AggeHandle
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge.run import neg_sample

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
import numpy as np
from math import ceil, floor


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

    def get_training_data(self, subject, sample_id):
        self.new_sample = sample_id != self.prev_sample_id
        self.df_train, self.df_test = read_data(subject)
        self.df_train.drop(columns=['usertag'], inplace=True)
        self.df_test.drop(columns=['usertag'], inplace=True)

        if self.new_sample:
            if subject in {'1458', '3386'}:
                self._df_test = neg_sample(self.df_test, 0.2)
                self._df_train = neg_sample(self.df_train, 0.2)

        return self.df_train


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
                self._df_test = neg_sample(self.df_test, 0.2)
                self._df_train = neg_sample(self.df_train, 0.5)

        cols_int = ['click', 'weekday', 'hour', 'timestamp', 'logtype', 'region', 'region', 'slotwidth', 'slotheight', 'slotheight', 'bidprice', 'payprice', 'payprice', 'event', 'slotprice_bucket']

        cols = ['weekday', 'hour',  # 'timestamp',
                'useragent', 'region', 'city', 'adexchange',
                'slotwidth', 'slotheight',
                'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                'creative',  # 'bidprice', #'payprice',
                'keypage', 'advertiser']

        dense_features = [item for item in cols if item in cols_int]
        sparse_features = [item for item in cols if item not in dense_features]

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

    def get_data_deepfm(self, subject, sample_id):

        self.new_subject = subject != self.prev_subject
        self.new_sample = sample_id != self.prev_sample_id

        self.prev_subject = subject
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
                self._df_test = neg_sample(self.df_test, 0.4)
                self._df_train = neg_sample(self.df_train, 0.4)

        cols_int = ['click', 'weekday', 'hour', 'timestamp', 'logtype', 'region', 'region', 'slotwidth', 'slotheight', 'slotheight', 'bidprice', 'payprice', 'payprice', 'event', 'slotprice_bucket']

        cols = ['weekday', 'hour',  # 'timestamp',
                'useragent', 'region', 'city', 'adexchange',
                'slotwidth', 'slotheight',
                'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                'creative',  # 'bidprice', #'payprice',
                'keypage', 'advertiser']

        dense_features = cols #[item for item in cols if item in cols_int]
        sparse_features = cols #[item for item in cols if item not in dense_features]

        for data in [self._df_train, self._df_test]:
            dnn_feature_columns_list = []
            linear_feature_columns_list = []
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
                
        X = self._df_train[self._df_train.columns[self._df_train.columns!='click']]
        y = self._df_train.click.to_numpy().astype('float64')
        X_test, y_test = self._df_test[self._df_test.columns[self._df_test.columns!='click']], self._df_test.click.to_numpy().astype('float64')
        
        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.15, stratify=y)
        X_train, y_train = fix_train_click_distribution(X_train, y_train)

        print(f'TRAIN CLICKS: {np.sum(y_train==1)}/{np.sum(y_train==1)+np.sum(y_train==0)}')
        print(f'VALI CLICKS: {np.sum(y_vali==1)}/{np.sum(y_vali==1)+np.sum(y_vali==0)}')
        print(f'')

        train_model_input = {name: X_train[name] for name in feature_names}
        test_model_input = {name: X_test[name] for name in feature_names}
        vali_model_input = {name: X_vali[name] for name in feature_names}

        data = {
            'X_train': train_model_input, 
            'X_test': test_model_input, 
            'X_vali': vali_model_input, 
            'y_train': y_train, 
            'y_test': y_test, 
            'y_vali': y_vali}

        return data, dnn_feature_columns_list, linear_feature_columns_list

def fix_train_click_distribution(df, y):
    df['click'] = y
    trues_df = df[df.click == 1]
    nopes_df = df[df.click == 0]
    print(trues_df.shape)
    print(nopes_df.shape)
    nth_row = int(ceil(nopes_df.shape[0]/trues_df.shape[0])) +1
    print(nth_row)
    print(nopes_df[::nth_row].shape)
    diff = nopes_df[::nth_row].shape[0] - trues_df.shape[0] 
    print(nopes_df[::nth_row].shape, trues_df[-diff:].shape)
    nopes_df[::nth_row] = trues_df[-diff:]

    return nopes_df[nopes_df.columns[nopes_df.columns!='click']], nopes_df.click.to_numpy().astype('float64')
    