import pandas as pd
import numpy as np
import os

def load_SWaT(home_dir):
    print("Reading SWaT...")
    print(f"current location: {os.getcwd()}")
    print(f"home dir: {home_dir}")
    SWAT_TRAIN_PATH = os.path.join(home_dir, './data/SWaT/SWaT_Dataset_Normal_v0.csv')
    SWAT_TEST_PATH = os.path.join(home_dir, './data/SWaT/SWaT_Dataset_Attack_v0.csv')
    df_train = pd.read_csv(SWAT_TRAIN_PATH, index_col = 0)
    df_test = pd.read_csv(SWAT_TEST_PATH, index_col = 0)

    def process_df(df):
        for var_index in [item for item in df.columns if item != 'Normal/Attack']:
            df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
        df.reset_index(drop=True, inplace=True)
        df.fillna(method='ffill', inplace=True)

        x = df.values[:,:-1].astype(np.float32)
        y = (df['Normal/Attack']=='Attack').to_numpy().astype(int)
        return x, y

    train_X, train_y = process_df(df_train)
    test_X, test_y = process_df(df_test)
    print("Loading complete.")
    return train_X, train_y, test_X, test_y
