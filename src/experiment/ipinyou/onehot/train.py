from experiment.ipinyou.onehot.encoder import MyOneHotEncoder

def train_encoder(__df_train):
    ohe = MyOneHotEncoder(cols=['weekday', 'hour',  # 'timestamp',
                    'useragent', 'region', 'city', 'adexchange',
                    'slotwidth', 'slotheight',
                    'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                    'creative',  # 'bidprice', #'payprice',
                    'keypage', 'advertiser']).fit(__df_train)

    return ohe