import os, time
import pandas as pd

from experiment.ipinyou.CONST import IPINYOU_DATA_DIR


def read_df(f):
    dtypes = {
        'click': 'Int8',
        'weekday': 'Int8',
        'hour': 'Int8',
        'bidid': 'string',
        'timestamp': 'Int64',
        'logtype': 'Int8',
        'ipinyouid': 'string',
        'useragent': 'string',
        'IP': 'string',
        'region': 'Int16',
        'city': 'Int16',
        'adexchange': 'string',
        'domain': 'string',
        'url': 'string',
        'urlid': 'string',
        'slotid': 'string',
        'slotwidth': 'Int16',
        'slotheight': 'Int16',
        'slotvisibility': 'string',
        'slotformat': 'string',
        'slotprice': 'Int64',
        'creative': 'string',
        'bidprice': 'Int64',
        'payprice': 'Int64',
        'keypage': 'string',
        'advertiser': 'Int16',
        'usertag': 'object'
    }

    df = pd.read_csv(f, dtype=dtypes, sep='\t')

    print('loaded df with columns = {}'.format(df.columns))

    return df

def read_adv(adv, set_type='test'):
    target = IPINYOU_DATA_DIR + '/make-ipinyou-data/{}/{}.log.txt'.format(adv, set_type)
    if not os.path.isfile(target):
        print('skip {}'.format(target))
        return None

    print('loading {} ... '.format(target))

    return read_df(target)


def read_advs(advs, set_type='test'):
    return pd.concat([read_adv(adv, set_type) for adv in advs], ignore_index=True)


def read_data(suject):
    # df = read_advs(all_adv, set_type='test')
    df_test = read_advs([suject], set_type='test')
    df_train = read_advs([suject], set_type='train')

    df_test.usertag = df_test.usertag.str.split(',')
    df_train.usertag = df_train.usertag.str.split(',')

    # should we fillna or keep unknown as any?
    def fill_na(df, usertag_replacement=[]):
        df.adexchange.fillna('unknown_adex', inplace=True)
        df.domain.fillna('unknown_domain', inplace=True)
        df.url.fillna('unknown_url', inplace=True)
        df.urlid.fillna('unknown_urlid', inplace=True)
        df.keypage.fillna('unknown_keypage', inplace=True)

        for row in df.loc[df['usertag'].isna(), 'usertag'].index:
            df.at[row, 'usertag'] = usertag_replacement

    fill_na(df_train)
    fill_na(df_test)

    print('na columns: {}'.format([column for column in df_train.columns if sum(df_train[column].isna()) > 0]))

    df_train['event'] = 1
    df_test['event'] = 1

    df_train['slotprice_bucket'] = df_train.slotprice // 6
    df_test['slotprice_bucket'] = df_test.slotprice // 6

    return df_train, df_test
