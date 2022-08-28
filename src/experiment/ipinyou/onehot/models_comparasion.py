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
    data_dict = {'2259':{'DeepFM': {'auc': [], 'delta': []}, 
                        'DCN': {'auc': [], 'delta': []}, 
                        'WDL': {'auc': [], 'delta': []},
                        'LogisticRegression': {'auc': [], 'delta': []}}, 
                '3358':{'DeepFM': {'auc': [], 'delta': []}, 
                        'DCN': {'auc': [], 'delta': []}, 
                        'WDL': {'auc': [], 'delta': []},
                        'LogisticRegression': {'auc': [], 'delta': []}}}

    for file in glob.glob(f'{path}/*'):
        sampler = file.split('_')[3]
        data = file.split('_')[4]
        df = pd.read_csv(file)
        start_dt = datetime.datetime.strptime(df.datetime_start.iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        finish_dt = datetime.datetime.strptime(df.datetime_complete.iloc[-1], '%Y-%m-%d %H:%M:%S.%f')
        delta = finish_dt - start_dt

        data_dict[data][sampler]['auc'] += [df.value.mean()]  
        data_dict[data][sampler]['delta'] += [delta.seconds]

    print(data_dict)

    # aucs
    auc_deepfm_2259 = data_dict['2259']['DeepFM']['auc']
    auc_dcn_2259 = data_dict['2259']['DCN']['auc']
    auc_wdl_2259 = data_dict['2259']['WDL']['auc']
    auc_lr_2259 = data_dict['2259']['LogisticRegression']['auc']
    auc_deepfm_3358 = data_dict['3358']['DeepFM']['auc']
    auc_dcn_3358 = data_dict['3358']['DCN']['auc']
    auc_wdl_3358 = data_dict['3358']['WDL']['auc']
    auc_lr_3358 = data_dict['3358']['LogisticRegression']['auc']

    #deltas
    delta_deepfm_2259 = data_dict['2259']['DeepFM']['delta']
    delta_dcn_2259 = data_dict['2259']['DCN']['delta']
    delta_wdl_2259 = data_dict['2259']['WDL']['delta']
    delta_lr_2259 = data_dict['2259']['LogisticRegression']['delta']
    delta_deepfm_3358 = data_dict['3358']['DeepFM']['delta']
    delta_dcn_3358 = data_dict['3358']['DCN']['delta']
    delta_wdl_3358 = data_dict['3358']['WDL']['delta']
    delta_lr_3358 = data_dict['3358']['LogisticRegression']['delta']
    

    aucs_deepfm = [average_lst(auc_deepfm_3358), average_lst(auc_deepfm_2259)]
    aucs_dcn = [average_lst(auc_dcn_3358), average_lst(auc_dcn_2259)]
    aucs_wdl = [average_lst(auc_wdl_3358), average_lst(auc_wdl_2259)]
    aucs_lr = [average_lst(auc_lr_3358), average_lst(auc_lr_2259)]

    delta_deepfm = [average_lst(delta_deepfm_3358), average_lst(delta_deepfm_2259)]
    delta_dcn = [average_lst(delta_dcn_3358), average_lst(delta_dcn_2259)]
    delta_wdl = [average_lst(delta_wdl_3358), average_lst(delta_wdl_2259)]
    delta_lr = [average_lst(delta_lr_3358), average_lst(delta_lr_2259)]

    aucs = aucs_deepfm + aucs_dcn + aucs_wdl + aucs_lr
    deltas = delta_deepfm + delta_dcn + delta_wdl + delta_lr
    model = ['DeepFM', 'DeepFM', 'DCN', 'DCN', 'WDL', 'WDL', 'Regresja logistyczna', 'Regresja logistyczna']
    data_set = ['3358', '2259', '3358', '2259', '3358', '2259', '3358', '2259']

    sns.set_theme()
    df = pd.DataFrame({'AUC': aucs, 'Delta': deltas, 'Model': model, 'Zbiór danych': data_set}, columns = ['AUC', 'Delta', 'Model', 'Zbiór danych'])
    plt.rcParams['figure.figsize'] = (10, 5)
    sns.barplot(x='Zbiór danych', y ='AUC', hue='Model', data=df)
    plt.ylim(0.6, 0.76)
    plt.title('Prównanie architektur - AUC')
    plt.show()

    sns.barplot(x='Zbiór danych', y ='Delta', hue='Model', data=df)
    plt.title('Prównanie architektur - Czas ewaluacji')
    plt.show()
    '''
        names = ['3358', '2259']
    x_axis = np.arange(len(names))
    plt.bar(x_axis-0.30, aucs_deepfm, label = 'DeepFM', width=0.3)
    plt.bar(x_axis-0.1, aucs_dcn, label = 'Cross&Deep', width=0.3)
    plt.bar(x_axis+0.1, aucs_dcn, label = 'Wide&Deep', width=0.3)
    #plt.bar(x_axis+0.3, aucs_lr, label = 'Regresja logistyczna', width=0.2)
    plt.ylim(0.55, 0.8)
    plt.xticks(x_axis, names)
    plt.title('Porówanie średnich wyników AUC dla modeli')
    plt.ylabel('AUC')
    plt.xlabel('Kampania reklamowa')
    plt.legend()
    plt.show()

    names = ['3358', '2259']
    x_axis = np.arange(len(names))
    plt.bar(x_axis-0.30, delta_deepfm, label = 'DeepFM', width=0.3)
    plt.bar(x_axis-0.1, delta_dcn, label = 'Cross&Deep', width=0.3)
    plt.bar(x_axis+0.1, delta_dcn, label = 'Wide&Deep', width=0.3)
    #plt.bar(x_axis+0.3, delta_lr, label = 'Regresja logistyczna', width=0.2)
    plt.xticks(x_axis, names)
    plt.title('Porówanie średnich czasów dla modeli')
    plt.ylabel('Czas')
    plt.xlabel('Kampania reklamowa')
    plt.legend()
    plt.show()
    
    '''




    '''
    auc_cmaes_2259 = [0.66273, 0.65385, 0.6673600000000001]
        auc_tpe_2259 = [0.6606266666666667, 0.6537, 0.6552566666666667]
        auc_cmaes_3358 = [0.7495, 0.7513433333333335, 0.7523366666666667]
        auc_tpe_3358 = [0.7485766666666668, 0.7343999999999999, 0.75593]

        delta_cmaes_2259 = [22147, 6893, 13008]
        delta_tpe_2259 = [18070, 5214, 6972]
        delta_cmaes_3358 = [20215, 17059, 37583]
        delta_tpe_3358 = [20702, 209, 18370]

        delta_cmaes = [average_lst(delta_cmaes_3358), average_lst(delta_cmaes_2259)] 
        delta_tpe = [average_lst(delta_tpe_3358), average_lst(delta_tpe_2259)] 

        aucs_cmaes = [average_lst(auc_cmaes_3358), average_lst(auc_cmaes_2259)] 
        aucs_tpe = [average_lst(auc_tpe_3358), average_lst(auc_tpe_2259)] 

        names = ['3358', '2259']
        x_axis = np.arange(len(names))
        plt.bar(x_axis-0.2, aucs_cmaes, label = 'CmaEs', width=0.4)
        plt.bar(x_axis+0.2, aucs_tpe, label = 'TPE', width=0.4)
        plt.ylim(0.6, 0.8)
        plt.xticks(x_axis, names)
        plt.title('Porówanie średnich wyników AUC dla metod próbkowania')
        plt.ylabel('AUC')
        plt.xlabel('Kampania reklamowa')
        plt.legend()
        plt.show()

        names = ['3358', '2259']
        x_axis = np.arange(len(names))
        plt.bar(x_axis-0.2, delta_cmaes, label = 'CmaEs', width=0.4)
        plt.bar(x_axis+0.2, delta_tpe, label = 'TPE', width=0.4)
        plt.xticks(x_axis, names)
        plt.title('Porówanie średnich czasów dla metod próbkowania')
        plt.ylabel('Czas')
        plt.xlabel('Kampania reklamowa')
        plt.legend()
        plt.show()




    '''
    