import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

from commons.utils import EarlyStopping
import os
from tqdm import tqdm as tqdm_plain
from tqdm.notebook import tqdm_notebook
from tqdm.gui import tqdm_gui

from exp_helpers.exp import Trainer, Tester


class LSTMAE_Trainer(Trainer):
    def __init__(self, args, model, train_loader, loss_fn, optimizer, scheduler = None):
        super(LSTMAE_Trainer, self).__init__(args, model, train_loader, loss_fn, optimizer, scheduler)

    def _process_batch(self, batch_data):
        batch_data = batch_data[0].to(self.args.device)
        predict_values = self.model(batch_data)

        self.optimizer.zero_grad()
        loss = self.loss_fn(batch_data, predict_values)
        loss.backward()
        self.optimizer.step()

        return loss


class LSTMAE_Tester(Tester):
    def __init__(self, args, model, train_loader, test_loader):
        super(LSTMAE_Tester, self).__init__(args, model, train_loader, test_loader)
        loss_fn = F.mse_loss

        self.train_loss_list = self.get_loss_list(self.train_iterator, loss_fn)
        self.test_loss_list = self.get_loss_list(self.test_iterator, loss_fn)
        self.mean = np.mean(self.train_loss_list, axis=0)
        self.std = np.cov(self.train_loss_list.T)

    def get_anomaly_score(self):
        anomaly_scores = []
        for item in self.test_loss_list:
            x = (item - self.mean)
            score = np.matmul(np.matmul(x, self.std), x.T)
            anomaly_scores.append(score)

        print(f"=== Anomaly statistics ===\n"
              f"Total: {len(anomaly_scores)}\n"
              f"mean[{np.mean(anomaly_scores)}], "
              f"median[{np.median(anomaly_scores)}], "
              f"min[{np.min(anomaly_scores)}], "
              f"max[{np.max(anomaly_scores)}]")
        return anomaly_scores

    def get_loss_list(self, dataloader, loss_fn):
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch_data in dataloader:
                batch_data = batch_data[0].to(self.args.device)
                predict_values = self.model(batch_data)

                loss = loss_fn(batch_data, predict_values, reduce=False)
                loss = loss.mean(dim=1).cpu().numpy()
                loss_list.append(loss)

        loss_list = np.concatenate(loss_list, axis=0)
        return loss_list
