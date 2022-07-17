import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os

if __name__ == "__main__":

    #read global data
    data = {}
    auc_dic = {}
    path = './mt_data/global/'
    
    for file in glob.glob(f'{path}*thread*'):

        try:
            threads_number = int(file[-2:])
        except:
            threads_number = int(file[-1])

        data[threads_number] = pd.read_csv(file)['1'].values[0]

        #read local data
        path_2 = './mt_data/'
        sum = 0
        counter = 0 
        print(threads_number)
        for experiment in glob.glob(f'{path_2}threads_{threads_number}/*thread*'):
            data_all = pd.read_csv(experiment)
            data_auc = float(data_all[data_all['0'] == 'BEST_TEST_AUC']['1'].values[0])
            sum += data_auc
            counter += 1
        auc_dic[threads_number] = sum/counter

    print(data)
    print(auc_dic)