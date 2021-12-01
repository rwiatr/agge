from sklearn.preprocessing import OneHotEncoder
import numpy as np

class MyOneHotEncoder():

    def __init__(self, cols):
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.cols = cols

    def fit(self, df):
        self.enc.fit(df[self.cols])
        return self

    def transform(self, df):
        results = []
        for col in self.cols:
            result = self.enc.transform(df[col])
            results += result

        return np.concat(results, axis=1)


    