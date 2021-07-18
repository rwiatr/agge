import json

import pandas as pd
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge2.train import train_estimators, train_lr_and_show_results
from experiment.measure import ProcessMeasure
import os


def _generate_space(generators):
    first_generator = generators[0]
    other_generators = generators[1:]

    if len(other_generators) == 0:
        for x in first_generator:
            yield x,
        return

    other = list(_generate_space(other_generators))
    for _first in first_generator:
        for _other in other:
            yield (_first,) + _other


def generate_space(generators, start=0, end=None):
    space = list(enumerate(list(_generate_space(generators))))
    if end is None:
        return space[start:]
    return space[start:end]


def neg_sample(df, ratio):
    clicks = df[df.click == 1]
    not_clicks = df[df.click == 0]
    return pd.concat([clicks, not_clicks.sample(int(df.shape[0] * ratio))], ignore_index=True)


if __name__ == '__main__':
    measure = ProcessMeasure()
    experiments = generate_space([
        # advertiser ids
        ['3476', '2259', '2261', '2821', '2997', '1458', '3358', '3386', '3427'],
        #         ['3358', '3427'],
        # bins
        # [1, 5, 10, 50, 150, 300],
        [1, 5, 10, 50, 150, 300],
        # re-runs
        # list(range(3)),
        list(range(5)),
        # bin type
        ['qcut'],
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0)
    print(experiments)
    prev_subject = None
    df_train, df_test = (None, None)

    reuse = True

    for experiment_id, (subject, bins, attempt, bin_type) in experiments:

        if reuse and os.path.isfile('test.pckl') and os.path.isfile('train.pckl'):
            _df_train = pd.read_pickle('train.pckl')
            _df_test = pd.read_pickle('test.pckl')
        else:
            if subject != prev_subject:
                df_train, df_test = read_data(subject)
                df_train.drop(columns=['usertag'], inplace=True)
                df_test.drop(columns=['usertag'], inplace=True)

            prev_subject = subject

            test_sampling = float(df[(df.advertiser == int(subject)) & (df.type == 'test')]['ctr']) * 2
            train_sampling = float(df[(df.advertiser == int(subject)) & (df.type == 'train')]['ctr']) * 2
            _df_test = neg_sample(df_test, test_sampling).sample(frac=0.1)
            _df_train = neg_sample(df_train, train_sampling).sample(frac=0.1)
            #             if subject in {'1458', '3386'}:
            #                 _df_test = neg_sample(df_test, 0.0005)
            #                 _df_train = neg_sample(df_train, 0.0005)
            #             else:
            #                 _df_test = neg_sample(df_test, 0.002)
            #                 _df_train = neg_sample(df_train, 0.002)

            _df_train.to_pickle('train.pckl')
            _df_test.to_pickle('test.pckl')

        __df_train, __df_test, estimators = train_estimators(_df_train, _df_test, None,
                                                             normalize_column="minmax",
                                                             bins=bins,
                                                             bin_type=bin_type)

        # experiment_output_K_L_P_ES
        fname = 'experiment_output_K=A_L=2a_P=0_ES=01.json'
        new_path = '2a'
        C = None
        norm_type = None
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + "_" + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        C = 10
        norm_type = 'L2'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        norm_type = 'L2+'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2+'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2+_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        C = 2
        norm_type = 'L2'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        norm_type = 'L2+'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2+'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2+_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        C = 15
        norm_type = 'L2'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

        norm_type = 'L2+'
        measure.set_suffix('_None_f={}_b={}_bt={}_r=L2+'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C) + '_L2+_' + str(bins),
                                       measure=measure, C=C,
                                       norm_type=norm_type, cnt_vec=None, f_name=fname, new_path=new_path)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))
    print('-------------------------------- RESULT --------------------------------')
    measure.print()