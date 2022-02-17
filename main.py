import argparse
import torch
import os
from torch.utils.data import Dataset

from data.load_data import load_data, load_anomaly_intervals
from data.dataset import get_dataset
from models.utils import prepare_model, prepare_loss_fn

from exp_helpers.exp_LSTMAE import LSTMAE_Trainer, LSTMAE_Tester
from exp_helpers.exp_OmniAnomaly import OmniAnomaly_Trainer, OmniAnomaly_Tester
from exp_helpers.exp_USAD import USAD_Trainer, USAD_Tester

# 1. Argparse
config = ""

print("=" * 30)
print(f"Parsing arguments...")
model_list = ["LSTMAE", "OmniAnomaly", "USAD"]
dataset_list = ["SWaT"]

parser = argparse.ArgumentParser(description='[TSAD] Time Series Anomaly Detection')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name, options: {dataset_list}")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required = False, default = 3, help=f"window size")
parser.add_argument("--epochs", type=int, required=False, default = 30, help=f"epochs to run")
parser.add_argument("--use_tqdm", type=str2bool, required=False, default = False, help="whether to use tqdm")
parser.add_argument("--tqdmopt", type=str, required=False, default = "plain", help=f"tqdm display options: plain, notebook, gui")
parser.add_argument("--load_pretrained", type=str2bool, required=False, default = False, help=f"whethre to load pretrained version")

## subparser of models
subparser = parser.add_subparsers(dest='model')

### LSTMAE
LSTMAE_parser = subparser.add_parser('LSTMAE')
LSTMAE_parser.add_argument("--latent_dim", type=int, required=True, default=128, help=f"LSTM hidden dim")
LSTMAE_parser.add_argument("--num_layers", type=int, required=True, default=1, help=f"Num of hidden layer")

### Omnianomaly
Omnianomaly_parser = subparser.add_parser("OmniAnomaly")
Omnianomaly_parser.add_argument("--latent_dim", type=int, required=True, default=128, help=f"LSTM hidden dim")
Omnianomaly_parser.add_argument("--num_layers", type=int, required=True, default=1, help=f"Num of hidden layer")
Omnianomaly_parser.add_argument("--z_dim", type=int, required=True, default=3, help=f"Num of hidden layer")
Omnianomaly_parser.add_argument("--dense_dim", type=int, required=True, default=10, help=f"Num of hidden layer")


### others
args = parser.parse_args()

config += f"_dataset_{args.dataset}" \
          f"_batch_size_{args.batch_size}" \
          f"_lr_{args.lr}" \
          f"_window_size_{args.window_size}" \

if args.model == "LSTMAE":
    config += f"_model_{args.model}" \
              f"_latent_dim_{args.latent_dim}" \
              f"_num_layers_{args.num_layers}"
elif args.model == "OmniAnomaly":
    config += f"_model_{args.model}" \
              f"_latent_dim_{args.latent_dim}" \
              f"_num_layers_{args.num_layers}" \
              f"_z_dim_{args.z_dim}" \
              f"_dense_dim_{args.dense_dim}"

args.config = config

os.makedirs(os.path.join("models", "checkpoints", f"{args.config}"), exist_ok=True)
args.model_path = os.path.join("models", "checkpoints", f"{args.config}", "model.pt")

os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", args.model), exist_ok=True)
os.makedirs(os.path.join("results", args.model, args.config), exist_ok=True)
args.result_path = os.path.join("results", args.model, args.config)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("done.")

# 2. Data
print("=" * 30)
print(f"preparing {args.dataset} dataset ...")
train_x, train_y, test_x, test_y = load_data(args.dataset)

train_dataset = get_dataset(train_x, train_y, window_size = args.window_size, dataset_type=args.dataset)
test_dataset = get_dataset(test_x, test_y, window_size = args.window_size, dataset_type=args.dataset)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_dataset,
                 batch_size=args.batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False)

args.input_dim = train_x.shape[1]

# 3. Model
print("=" * 30)
print(f"setting model to {args.device}...")
model = prepare_model(args)
model.to(args.device)
print("done.")

# 4. train
print("=" * 30)
print("Training...")

#args.epochs = 1
#args.load_pretrained = True

trainers = {
    "LSTMAE": LSTMAE_Trainer,
    "OmniAnomaly": OmniAnomaly_Trainer,
    "USAD": USAD_Trainer,
}

if args.load_pretrained is True:
    print(f"loading pretrained model at {args.model_path}...")
    best_model = model
    best_model.load_state_dict(torch.load(args.model_path))
    best_model.to(args.device)
else:
    print("start training...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr = args.lr)
    loss_fn = prepare_loss_fn(args)

    trainer = trainers[args.model](
        args=args,
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    best_model = trainer.train()

print("done.")

# 5. test
print("=" * 30)
print("Testing...")

testers = {
    "LSTMAE": LSTMAE_Tester,
    "OmniAnomaly": OmniAnomaly_Tester,
    "USAD": USAD_Tester,
}

tester = testers[args.model](
    args = args,
    model = best_model,
    train_loader = train_loader,
    test_loader = test_loader,
)
anomaly_scores = tester.get_anomaly_score()
intervals = load_anomaly_intervals(anomaly_labels = test_y, window_size = args.window_size)
best_threshold = tester.regular_thresholding(test_y, anomaly_scores, intervals, num_candidates = 100)

print("done.")