import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_pickle("results.pickle").explode('value')
    df.value = df.value.astype(int)
    print(df.value)
    print(df.columns)
    print(df.dtypes)
    print(df.measure_type.unique())
    print(df.algorithm.unique())

    df = df[((df.algorithm == "MLP-v0") & (df.alpha == "0.0001")) |
       ((df.algorithm == "SKLearn-MLP") & (df.alpha == "0.01"))]
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