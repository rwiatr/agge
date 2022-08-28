from tkinter import CENTER
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import datetime
import collections
import seaborn as sns

if __name__ == '__main__':
    #AUCs = [0.7586, 0.7607, 0.7659, 0.7491, 0.7658, 0.7552, 0.7657, 0.7309] # wyniki dla 3358 TPE
    #AUCs_bis = [0.7658, 0.7552, 0.7657, 0.7309] # wyniki dla 3358 CMAES
    AUCs = [0.6645, 0.6704, 0.6592, 0.6477, 0.6554, 0.6656, 0.6730, 0.6518]
    #AUCs = [0.6645, 0.6704, 0.6592, 0.5932] # wyniki dla 2259 TPE
    #AUCs_bis = [0.6554, 0.6656, 0.6730, 0.5790] #wyniki dla 2259 CMAES
    names = ['DeepFM', 'Deep&Cross', 'Wide&Deep', 'Regresja logistyczna', 'DeepFM', 'Deep&Cross', 'Wide&Deep', 'Regresja logistyczna']
    samplers = ['TPE', 'TPE', 'TPE', 'TPE', 'CmaEs', 'CmaEs', 'CmaEs', 'CmaEs']

    df_data = pd.DataFrame({'Model': names,'AUC': AUCs, 'Sampler': samplers})
    print(df_data)

    #sns.set_theme(style="whitegrid")
    sns.set_theme()
    plt.rcParams['figure.figsize'] = (10, 5)
    sns.barplot(x='Model', y='AUC', hue='Sampler', data = df_data)
    plt.ylim(0.60, 0.68)
    plt.title('Wyniki AUC dla najlepszego zbioru hiperparametrów')
    plt.legend()
    plt.show()

    '''
    x_axis = np.arange(len(names))
    plt.bar(x_axis-0.2, AUCs, label = 'TPE', width=0.4)
    plt.bar(x_axis+0.2, AUCs_bis, label = 'CmaEs', width=0.4)
    plt.ylim(0.7, 0.8)
    plt.xticks(x_axis, names)
    plt.title('Zbiór 3358')
    plt.ylabel('AUC')
    plt.xlabel('Model')
    plt.legend()
    plt.show()
    '''
