import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from experiment.ipinyou.onehot.data_manager import DataManager


dt_mng = DataManager()

data = dt_mng.get_training_data('1458', 1)

print(data['slotvisibility'])

## string cols: 
## int cols: 'click', 'weekday', 'hour', 'timestamp', 'logtype', 'region', 'region', 'slotwidth', 'slotheight', 'slotheight', 'bidprice', 'payprice', 'payprice', 'event', 'slotprice_bucket' 
    
cols_int = ['click', 'weekday', 'hour', 'timestamp', 'logtype', 'region', 'region', 'slotwidth', 'slotheight', 'slotheight', 'bidprice', 'payprice', 'payprice', 'event', 'slotprice_bucket', 'city', 'adexchange', 'slotvisibility', 'slotformat']

cols = ['weekday', 'hour',  # 'timestamp',
                'useragent', 'region', 'city', 'adexchange',
                'slotwidth', 'slotheight',
                'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                'creative',  # 'bidprice', #'payprice',
                'keypage', 'advertiser']



dense_features = [item for item in cols if item in cols_int]

print(data[dense_features])

sparse_features = [item for item in cols if item not in dense_features]

print(data[sparse_features])
