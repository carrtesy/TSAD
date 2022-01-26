import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from tqdm.gui import tqdm_gui
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List
import easydict

from model import OmniAnomaly

def load_data(is_train=True):
    if is_train:
        sensor_path = '../dataset/SWaT/SWaT_Dataset_Normal_v0.csv'
        df = pd.read_csv(sensor_path, index_col=0)
    else:
        sensor_path = '../dataset/SWaT/SWaT_Dataset_Attack_v0.csv'
        df = pd.read_csv(sensor_path, index_col=0)


    ## 데이터 Type 변경
    for var_index in [item for item in df.columns if item != 'Normal/Attack']:
        df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
    df.reset_index(drop=True, inplace=True)
    df.fillna(method='ffill', inplace=True)

    print(df.head())

    x = df.values[:,:-1].astype(np.float32)
    print(x)
    y = (df['Normal/Attack']=='Normal').to_numpy().astype(int)
    print(y)

    return x, y

tx, ty = load_data(True)
test_x, test_y = load_data(False)

n_train = int(tx.shape[0]*0.7)
train_x, train_y = tx[:n_train, :], ty[:n_train]
val_x, val_y = tx[n_train:, :], ty[n_train:]

x_min = np.min(train_x, 0, keepdims=True)
x_max = np.max(train_x, 0, keepdims=True)

## 설정 폴더
args = easydict.EasyDict({
    "batch_size": 128, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "input_size": train_x.shape[1], ## 입력 차원 설정
    "latent_size": 10, ## Hidden 차원 설정
    "output_size": train_x.shape[1], ## 출력 차원 설정
    "window_size" : 3, ## sequence Lenght
    "num_layers": 2,     ## LSTM layer 갯수 설정
    "learning_rate" : 0.001, ## learning rate 설정
    "max_iter" : 100000, ## 총 반복 횟수 설정
    'early_stop' : True,  ## valid loss가 작아지지 않으면 early stop 조건 설정
})


class SWaTDataset(Dataset):
    def __init__(self, x, y, x_min, x_max, window_size=1):
        super().__init__()
        t = (x_min != x_max).astype(np.float32)
        self.x = (x - x_min) / (x_max-x_min + 1e-5) * t
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return self.x.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return self.x[idx:idx+self.window_size], self.y[idx:idx+self.window_size]

def loss_function(x, pred_x, mu, sigma):
    MSE_loss = F.mse_loss(x, pred_x)
    KLD_loss = -0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma)
    return MSE_loss + KLD_loss

## 데이터셋으로 변환
train_dataset = SWaTDataset(train_x, train_y, x_min, x_max, window_size=args.window_size)
val_dataset = SWaTDataset(val_x, val_y, x_min, x_max, window_size=args.window_size)
test_dataset = SWaTDataset(test_x, test_y, x_min, x_max, window_size=args.window_size)


## Data Loader 형태로 변환
train_loader = torch.utils.data.DataLoader(
                 dataset=train_dataset,
                 batch_size=args.batch_size,
                 shuffle=True)
valid_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=False)
test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False)


model = OmniAnomaly(
    in_dim=args.input_size,
    hidden_dim=500,
    z_dim=3,
    dense_dim=500,
    out_dim=args.input_size
)

model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

## 반복 횟수 Setting
epochs = tqdm_gui(range(args.max_iter//len(train_loader)+1), leave=True)

## Training
best_loss = None
for epoch in epochs:
    model.train()
    optimizer.zero_grad()
    train_iterator = tqdm_gui(enumerate(train_loader), total=len(train_loader), desc="training", leave=True)

    for i, batch_data in train_iterator:
        batch_data = batch_data[0].to(args.device)
        predict_values, mu, logvar = model(batch_data)

        # Backward and optimize

        optimizer.zero_grad()
        loss = loss_function(batch_data, predict_values, mu, logvar)
        loss.backward()
        optimizer.step()

        train_iterator.set_postfix({
            "train_loss": float(loss),
        })

    model.eval()
    eval_loss = 0
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing", leave=True)
    with torch.no_grad():
        for i, batch_data in test_iterator:

            batch_data = batch_data[0].to(args.device)
            predict_values, mu, sigma = model(batch_data)
            loss = loss_function(batch_data, predict_values, mu, sigma)

            eval_loss += loss.mean().item()

            test_iterator.set_postfix({
                "eval_loss": float(loss),
            })

    eval_loss = eval_loss / len(test_loader)
    epochs.set_postfix({
         "Evaluation Score": float(eval_loss),
    })
    if best_loss is None or eval_loss < best_loss:
        best_loss = eval_loss
    else:
        if args.early_stop:
            print('early stop condition   best_loss[{}]  eval_loss[{}]'.format(best_loss, eval_loss))