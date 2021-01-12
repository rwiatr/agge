import pandas as pd
from experiment.ipinyou.load import read_data
from experiment.ipinyou.agge.train import train_estimators, train_lr_and_show_results
from experiment.measure import ProcessMeasure

# todo use from train package
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
        # ['1458', '3358', '3386', '3427', '3476', '2259', '2261', '2821', '2997'],
        ['2997'],
        # bins
        [1, 3, 5, 10, 50, 100, 200, 300],
        # re-runs
        list(range(5)),
        # bin type
        ['qcut']
    ],
        # starting experiment id (you can skip start=N experiments in case of error)
        start=0)
    print(experiments)
    prev_subject = None
    df_train, df_test = (None, None)
    for experiment_id, (subject, bins, attempt, bin_type) in experiments:

        if subject != prev_subject:
            df_train, df_test = read_data(subject)
            df_train.drop(columns=['usertag'], inplace=True)
            df_test.drop(columns=['usertag'], inplace=True)

        prev_subject = subject

        if subject in {'1458', '3386'}:
            _df_test = neg_sample(df_test, 0.2)
            _df_train = neg_sample(df_train, 0.2)
        else:
            _df_test = neg_sample(df_test, 0.5)
            _df_train = neg_sample(df_train, 0.5)

        __df_train, __df_test, estimators = train_estimators(_df_train, _df_test, None,
                                                             normalize_column="minmax",
                                                             bins=bins,
                                                             bin_type=bin_type, echo=False)
        C = 1
        measure.set_suffix('_None_f={}_b={}_bt={}'.format(__df_train.shape[1], bins, bin_type))
        lr = train_lr_and_show_results(__df_train, __df_test, None, subject + '_' + str(C), measure=measure, C=C)
        print('Done experiment id={}, adv={}, bins={}, attempt={}'.format(experiment_id, subject, bins, attempt))

    print('-------------------------------- RESULT --------------------------------')
    measure.print()
