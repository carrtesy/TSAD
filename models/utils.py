import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LSTMAE import LSTMAutoEncoder
from models.OmniAnomaly import OmniAnomaly
from models.USAD import USAD


def prepare_model(args):
    model = None
    if args.model == "LSTMAE":
        model_args = {
            "input_dim": args.input_dim,
            "latent_dim": args.latent_dim,
            "window_size": args.window_size,
            "num_layers": args.num_layers
        }
        model = LSTMAutoEncoder(**model_args)
    elif args.model == "OmniAnomaly":
        model_args = {
            "in_dim": args.input_dim,
            "hidden_dim": args.latent_dim,
            "z_dim": args.z_dim,
            "dense_dim": args.dense_dim,
            "out_dim": args.input_dim,
        }
        model = OmniAnomaly(**model_args)
    elif args.model == "USAD":
        model_args = {
            "input_size" : args.input_dim,
            "latent_space_size" : args.latent_dim,
        }
        model = USAD(**model_args)

    if model is None:
        raise ValueError("Model is not defined")
    return model

def prepare_loss_fn(args):
    loss_fn = None
    if args.model == "LSTMAE":
        loss_fn = lambda x, pred_x: F.mse_loss(x, pred_x)
    elif args.model == "OmniAnomaly":
        def loss_function(x, pred_x, mu, logvar):
            MSE_loss = F.mse_loss(x, pred_x, reduction="sum")
            KLD_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
            return MSE_loss + KLD_loss
        loss_fn = loss_function
    elif args.model == "USAD":
        def loss_function(Wt, Wt1p, Wt2p, Wt2dp, n):
            """
                :param Wt: ground truth sequence
                :param Wt1p: AE1 decoder output
                :param Wt2p: AE2 decoder output
                :param Wt2dp: AE1 encoder output => AE2 decoder
                :param n: Training epochs
            """
            loss_AE1 = (1 / n) * F.mse_loss(Wt, Wt1p) + (1 - (1 / n)) * F.mse_loss(Wt, Wt2dp)
            loss_AE2 = (1 / n) * F.mse_loss(Wt, Wt2p) - (1 - (1 / n)) * F.mse_loss(Wt, Wt2dp)
            return loss_AE1 + loss_AE2
        loss_fn = loss_function

    if loss_fn is None:
        raise ValueError("Loss function is not defined")
    return loss_fn