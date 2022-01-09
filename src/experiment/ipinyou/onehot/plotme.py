import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_pickle("results_0.pickle").explode('value')
    df.value = df.value.astype(float)
    df.n_iter_no_change = df.n_iter_no_change.astype(int)
    print(df.value)
    print(df.columns)
    print(df.dtypes)
    print(df.measure_type.unique())
    print(df.algorithm.unique())

    df = df[((df.algorithm == "MLP-v0") & (df.alpha == "0.0001")) |
       ((df.algorithm == "SKLearn-MLP") & (df.alpha == "0.0001")) | ((df.algorithm == 'DeepWide'))]
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
        df[(df.measure_type == "auc_test") & (df.subject == subject)].boxplot(by=["algorithm", "learning_rate_init"], column="n_iter_no_change")
        plt.xticks(rotation='vertical')
        plt.title("test " + subject)
        df[(df.measure_type == "auc_train") & (df.subject == subject)].boxplot(by=["algorithm", "learning_rate_init"], column="n_iter_no_change")
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
