import pandas as pd
import numpy as np
import os


def load_data(data_path):
    print(os.getcwd())
    df = pd.read_csv(data_path, index_col = 0)

    for var_index in [item for item in df.columns if item != 'Normal/Attack']:
        df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
    df.reset_index(drop=True, inplace=True)
    df.fillna(method='ffill', inplace=True)

    print(df.head())

    x = df.values[:,:-1].astype(np.float32)
    y = (df['Normal/Attack']=='Attack').to_numpy().astype(int)

    return x, y

def load_anomaly_intervals(anomaly_labels, window_size):
    window_y = []
    for i in range(len(anomaly_labels)):
        window_y.append(max(anomaly_labels[i:i + window_size]) == 1)

    intervals = []
    start = None
    for i, label in enumerate(window_y):
        if label:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(window_y) - 1))

    return intervals
