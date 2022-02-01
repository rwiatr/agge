import pandas as pd


def make_conjunction_features(df):
    conj_cols = []
    cols = [col for col in df.columns if col != 'click']

    series = []
    for col_a in df.columns:
        for col_b in df.columns:
            if col_a != 'click' and col_b != 'click':
                if col_a != col_b:
                    series.append(df[col_a].astype(str) + df[col_b].astype(str))
                    conj_cols.append(col_a + '_' + col_b + "__conj")
    series = pd.concat(series, axis=1, names=conj_cols)
    series.rename(columns={idx: name for idx, name in enumerate(conj_cols)}, inplace=True)

    df = pd.concat([df, series], axis=1)

    print(df.columns)

    return df, cols, conj_cols
