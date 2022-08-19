import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    #read global data
    data = {}
    auc_dic = {}
    

    global_times = [0 for x in range(9)]
    
    # read global times
    for file in glob.glob(f'{path}*thread*'):
        global_times[int(file.split('_')[5][0])] = pd.read_csv(file)['1'].values[0]

    auc_list = [0 for x in range(9)]
    for file in glob.glob('./optuna_data/threads*'):
        count = 0
        auc = 0
        for file_2 in glob.glob(file+'/*'):
            
            print(file_2.split('\\')[1][-1])
            auc += float(pd.read_csv(file_2)['1'][8])
            count += 1
        auc = auc/count
        auc_list[int(file_2.split('\\')[1][-1])] = auc

    auc_fixed = [auc_list[x] for x in [1, 2, 4, 8]]
    global_fixed = [global_times[x] for x in [1, 2, 4, 8]]

    times = global_fixed
    times = [time/times[0] for time in times]
    delta = times

    plt.rcParams['figure.figsize'] = (20, 5)
    plt.plot([1, 2, 4, 8], delta, label = 'delta')
    plt.ylabel('Wartość')
    plt.xlabel('Liczba wątków')

    plt.legend()
    plt.show()

    plt.plot([1, 2, 4, 8],auc_fixed,'g', label = 'AUC')
    plt.ylabel('Wartość AUC')
    plt.xlabel('Liczba wątków')
    plt.legend()
    plt.show()