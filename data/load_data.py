import pandas as pd
import numpy as np
import os

from load_data_submodule import load_SWaT

functions = {}
functions["SWaT"] = load_SWaT

def load_data(dataset_type = "SWaT"):
    return functions[dataset_type]()

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
