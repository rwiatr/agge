import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
import datetime
import collections
import seaborn as sns

if __name__ == '__main__':
    path = './plotme3'
    data_dict = {}

    for file in glob.glob(f'{path}/*'):
        print(file.split('_')[6])
        data_dict[int(file.split('_')[6])] = pd.read_csv(file)

    seconds_dict = {}
    val_dict = {}
    total_auc = 0.0
    for k, i in data_dict.items():


        start_dt = datetime.datetime.strptime(data_dict[k].start.iloc[0], '%c') # %Y-%m-%d %H:%M:%S.%f
        finish_dt = datetime.datetime.strptime(data_dict[k].finish.iloc[-1], '%c') # %Y-%m-%d %H:%M:%S.%f
        delta = finish_dt - start_dt

        seconds_dict[k] = delta.seconds
        val_dict[k] = data_dict[k].best_test_auc.mean()

    ordered_dict = collections.OrderedDict(sorted(seconds_dict.items()))
    ordered_dict_vals = collections.OrderedDict(sorted(val_dict.items()))
    df_data = pd.DataFrame(ordered_dict_vals.items(), columns = ['Liczba Wątków', 'AUC'])
    df_delta = pd.DataFrame(ordered_dict.items(), columns = ['Liczba Wątków', 'Delta'])
    sns.set_theme()
    plt.rcParams['figure.figsize'] = (20, 5)
    sns.set()
    sns.lineplot(data= df_data, x='Liczba Wątków', y='AUC', linewidth=6, label = 'AUC')
    plt.title('Optuna, rozpraszanie badania na wątki - AUC')
    plt.legend()
    plt.show()
    plt.title('Optuna, rozpraszanie badania na wątki - Delta')
    sns.lineplot(data= df_delta, x='Liczba Wątków', y='Delta', linewidth=6, label = 'Czas ewaluacji')
    plt.legend()
    plt.show()

    '''
        plt.rcParams['figure.figsize'] = (20, 5)
        plt.plot(ordered_dict.keys(), ordered_dict.values(), label = 'delta', linewidth=5)
        plt.ylabel('Wartość [s]')
        plt.xlabel('Liczba wątków')

        plt.legend()
        plt.show()

        plt.plot(ordered_dict_vals.keys(), ordered_dict_vals.values(), label = 'AUC', linewidth=5)
        plt.ylabel('Wartość')
        plt.xlabel('Liczba wątków')

        plt.legend()
        plt.show()

    '''


    '''

    data_dict = {}
    path = './plotme2'

    for file in glob.glob(f'{path}/*'):
        print(file.split('_')[3][-1])
        data_dict[file.split('_')[4][-1]] = pd.read_csv(file)

    seconds_dict = {}
    val_dict = {}
    total_auc = 0.0
    for k, i in data_dict.items():


        start_dt = datetime.datetime.strptime(data_dict[k].datetime_start.iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        finish_dt = datetime.datetime.strptime(data_dict[k].datetime_complete.iloc[-1], '%Y-%m-%d %H:%M:%S.%f')
        delta = finish_dt - start_dt

        seconds_dict[k] = delta.seconds
        val_dict[k] = data_dict[k].value.mean()

    print(seconds_dict)
    seconds_dict['4'] = 24575.974799394608
    val_dict['4'] = 0.77542

    seconds_dict['8'] = 21792.54026079178
    val_dict['8'] = 0.77116

    ordered_dict = collections.OrderedDict(sorted(seconds_dict.items()))
    ordered_dict_vals = collections.OrderedDict(sorted(val_dict.items()))


    plt.rcParams['figure.figsize'] = (20, 5)
    plt.plot(ordered_dict.keys(), ordered_dict.values(), label = 'delta')
    plt.ylabel('Wartość [s]')
    plt.xlabel('Liczba wątków')

    plt.legend()
    plt.show()

    plt.plot(ordered_dict_vals.keys(), ordered_dict_vals.values(), label = 'AUC')
    plt.ylabel('Wartość')
    plt.xlabel('Liczba wątków')

    plt.legend()
    plt.show()

    '''
 
    '''

    path = './plotme'
        data_dict = {}

        for file in glob.glob(f'{path}/*'):
            print(file.split('_')[3][-1])
            data_dict[file.split('_')[3][-1]] = pd.read_csv(file)

        seconds_dict = {}
        val_dict = {}
        total_auc = 0.0
        for k, i in data_dict.items():


            start_dt = datetime.datetime.strptime(data_dict[k].start.iloc[0], '%c')
            finish_dt = datetime.datetime.strptime(data_dict[k].finish.iloc[-1], '%c')
            delta = finish_dt - start_dt

            seconds_dict[k] = delta.seconds
            val_dict[k] = data_dict[k].best_test_auc.mean()

        ordered_dict = collections.OrderedDict(sorted(seconds_dict.items()))
        ordered_dict_vals = collections.OrderedDict(sorted(val_dict.items()))
        plt.rcParams['figure.figsize'] = (20, 5)
        plt.plot(ordered_dict.keys(), ordered_dict.values(), label = 'delta')
        plt.ylabel('Wartość [s]')
        plt.xlabel('Liczba wątków')

        plt.legend()
        plt.show()

        plt.plot(ordered_dict_vals.keys(), ordered_dict_vals.values(), label = 'AUC')
        plt.ylabel('Wartość')
        plt.xlabel('Liczba wątków')

        plt.legend()
        plt.show()
    '''





    '''
    df = pd.read_pickle("results_0.pickle").explode('value')
    df.value = df.value.astype(float)
    df.n_iter_no_change = df.n_iter_no_change.astype(int)
    print(df.value)
    print(df.columns)
    print(df.dtypes)
    print(df.measure_type.unique())
    print(df.algorithm.unique())

    df = df[((df.algorithm == "MLP-v0") & (df.alpha == "0.0001")) |
       ((df.algorithm == "SKLearn-MLP") & (df.alpha == "0.0001")) | ((df.algorithm == 'DeepWide') & (df.alpha == "0.0001"))]
    print(df)
    for subject in df.subject.unique():
        df[(df.measure_type == "auc_test") & (df.subject == subject)].boxplot(by="algorithm", column="value")
        plt.title("test " + subject)
        df[(df.measure_type == "auc_train") & (df.subject == subject)].boxplot(by="algorithm", column="value")
        plt.title("train " + subject)
    plt.show()

    for subject in df.subject.unique():
        df[(df.measure_type == "auc_test") & (df.subject == subject)].boxplot(by=["algorithm", "alpha"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject)
        df[(df.measure_type == "auc_train") & (df.subject == subject)].boxplot(by=["algorithm", "alpha"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("train " + subject)
    plt.show()

    for subject in df.subject.unique():
        df[(df.measure_type == "auc_test") & (df.subject == subject)].boxplot(by=["algorithm", "learning_rate_init"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject)
        df[(df.measure_type == "auc_train") & (df.subject == subject)].boxplot(by=["algorithm", "learning_rate_init"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("train " + subject)
    plt.show()

    for subject in df.subject.unique():

        df[(df.algorithm == "DeepWide") & (df.subject == subject)].boxplot(by=["measure_type"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject + " DeepWide")
        df[(df.algorithm == "MLP-v0") & (df.subject == subject)].boxplot(by=["measure_type"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject + " mlp")
        df[(df.algorithm == "SKLearn-MLP") & (df.subject == subject)].boxplot(by=["measure_type"], column="value")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject + " SKlearn")
    plt.show()
    '''
