import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import datetime
import collections
import seaborn as sns

def average_lst(lst):
    return sum(lst)/len(lst)

if __name__ == '__main__':
    path = './optuna_data/global/sampler_comparasion'
    data_dict = {'2259':{'CmaEsSampler': {'auc': [], 'delta': []}, 'TPESampler': {'auc': [], 'delta': []}}, '3358':{'CmaEsSampler':{'auc': [], 'delta': []}, 'TPESampler': {'auc': [], 'delta': []}}}

    for file in glob.glob(f'{path}/*'):
        sampler = file.split('_')[5]
        data = file.split('_')[4]
        df = pd.read_csv(file)
        start_dt = datetime.datetime.strptime(df.datetime_start.iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        finish_dt = datetime.datetime.strptime(df.datetime_complete.iloc[-1], '%Y-%m-%d %H:%M:%S.%f')
        delta = finish_dt - start_dt

        data_dict[data][sampler]['auc'] += [df.value.mean()]  
        data_dict[data][sampler]['delta'] += [delta.seconds]

    print(data_dict)

    auc_cmaes_2259 = data_dict['2259']['CmaEsSampler']['auc']
    auc_tpe_2259 = data_dict['2259']['TPESampler']['auc']
    auc_cmaes_3358 = data_dict['3358']['CmaEsSampler']['auc']
    auc_tpe_3358 = data_dict['3358']['TPESampler']['auc']

    delta_cmaes_2259 = data_dict['2259']['CmaEsSampler']['delta']
    delta_tpe_2259 = data_dict['2259']['TPESampler']['delta']
    delta_cmaes_3358 = data_dict['3358']['CmaEsSampler']['delta']
    delta_tpe_3358 = data_dict['3358']['TPESampler']['delta']

    delta_cmaes = [average_lst(delta_cmaes_3358), average_lst(delta_cmaes_2259)] 
    delta_tpe = [average_lst(delta_tpe_3358), average_lst(delta_tpe_2259)] 

    aucs_cmaes = [average_lst(auc_cmaes_3358), average_lst(auc_cmaes_2259)] 
    aucs_tpe = [average_lst(auc_tpe_3358), average_lst(auc_tpe_2259)] 

    aucs = aucs_tpe + aucs_cmaes
    delta = delta_tpe + delta_cmaes
    samplers = ['TPE', 'TPE', 'CmaEs', 'CmaEs']
    data_sets = ['3358', '2259', '3358', '2259']

    df = pd.DataFrame({'AUC': aucs, 'Delta': delta, 'Zbiór danych': data_sets, 'Sampler': samplers}, columns = ['AUC', 'Delta', 'Zbiór danych', 'Sampler'])
    print(df)
    sns.set_theme()
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.ylim(0.6, 0.8)
    sns.barplot(x='Zbiór danych', y='AUC', hue='Sampler', data = df)
    plt.title('Porównanie metod próbkowania - AUC')
    plt.show()

    sns.barplot(x='Zbiór danych', y='Delta', hue='Sampler', data = df)
    plt.title('Porównanie metod próbkowania - Czas ewaluacji')
    plt.show()


    '''
   
    names = ['3358', '2259']
    x_axis = np.arange(len(names))
    
    plt.bar(x_axis-0.2, aucs_tpe, label = 'TPE', width=0.4)
    plt.bar(x_axis+0.2, aucs_cmaes, label = 'CmaEs', width=0.4)
    plt.ylim(0.6, 0.8)
    plt.xticks(x_axis, names)
    plt.title('Porówanie średnich wyników AUC dla metod próbkowania')
    plt.ylabel('AUC')
    plt.xlabel('Kampania reklamowa')
    plt.legend()
    plt.show()

    names = ['3358', '2259']
    x_axis = np.arange(len(names))
    
    plt.bar(x_axis-0.2, delta_tpe, label = 'TPE', width=0.4)
    plt.bar(x_axis+0.2, delta_cmaes, label = 'CmaEs', width=0.4)
    plt.xticks(x_axis, names)
    plt.title('Porówanie średnich czasów dla metod próbkowania')
    plt.ylabel('Czas')
    plt.xlabel('Kampania reklamowa')
    plt.legend()
    plt.show() 
    
    
    '''




