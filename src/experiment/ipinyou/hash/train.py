from experiment.ipinyou.hash.encoder import HashFeatureEncoder


def train_encoder(__df_train, size):
    has_usertag = 'usertag' in __df_train.columns
    hfe = HashFeatureEncoder(
        hash_space=size,  # 513, #1013,
        columns=['weekday', 'hour',  # 'timestamp',
                 'useragent', 'region', 'city', 'adexchange',
                 'slotwidth', 'slotheight',
                 'slotvisibility', 'slotformat', 'slotprice_bucket',  # 'slotprice',
                 'creative',  # 'bidprice', #'payprice',
                 'keypage', 'advertiser']
        , array_columns=(['usertag'] if has_usertag else None)
    ).fit(__df_train)
    return hfe
