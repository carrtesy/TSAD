import pandas as pd
import numpy as np
import os


def load_data(data_path):
    df = pd.read_csv(data_path, index_col = 0)

    for var_index in [item for item in df.columns if item != 'Normal/Attack']:
        df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
    df.reset_index(drop=True, inplace=True)
    df.fillna(method='ffill', inplace=True)

    print(df.head())

    x = df.values[:,:-1].astype(np.float32)
    y = (df['Normal/Attack']=='Normal').to_numpy().astype(int)

    return x, y