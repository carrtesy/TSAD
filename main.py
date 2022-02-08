import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import load_data
from LSTM_ENC_DEC_AD.model import LSTMAutoEncoder
from Omnianomaly_pytorch.model import OmniAnomaly
from USAD.model import USAD
from data.dataset import get_dataset

model_list = ["LSTM_ENC_DEC", "OmniAnomaly", "USAD"]
dataset_list = ["SWaT"]

# 1. Argparse
parser = argparse.ArgumentParser(description='[TSAD] Time Series Anomaly Detection')

parser.add_argument("--model", type=str, required=True, default="LSTM_ENC_DEC", help=f"Model name, options: {model_list}")
parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name, options: {dataset_list}")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")

args = parser.parse_args()

# 2. Data
train_x, train_y, test_x, test_y = load_data(args.dataset)

train_dataset = get_dataset(train_x, train_y, window_size = 3, dataset_type=args.dataset)
test_dataset = get_dataset(test_x, test_y, window_size = 3, dataset_type=args.dataset)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_dataset,
                 batch_size=args.batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False)

# 3. Model
model_args = {
    "input_dim" : args.input_size,
    "latent_dim" : args.latent_size,
    "window_size" : args.window_size,
    "num_layers" : args.num_layers
} # temporary setting

models = {
    "LSTM_ENC_DEC": LSTMAutoEncoder(**model_args),
    #"OmniAnomaly": OmniAnomaly(**model_args),
    #"USAD": USAD(**model_args),
}

model = models[args.model]
model.to(args.device)

# 4. train
## implement training loop here ##
# model.run()
##########

# 5. test
## implement testing loop here ##
# model.test()
##########
