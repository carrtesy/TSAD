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


class Trainer:
    def __init__(self, args, model, train_loader, loss_fn, optimizer, scheduler = None):
        self.args = args
        print(args)
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.args.use_tqdm:
            tqdms = {
                "plain": tqdm_plain,
                "notebook": tqdm_notebook,
                "gui" : tqdm_gui,
            }
            self.tqdm = tqdms[args.tqdmopt]

        self.early_stopping = EarlyStopping(patience=3, path=self.args.model_path)

    def train(self):
        self.model.train()

        epochs = range(self.args.epochs) if not self.args.use_tqdm \
            else self.tqdm(range(self.args.epochs), leave=True)

        for epoch in epochs:
            train_loss = 0.0
            train_iterator = enumerate(self.train_loader) if not self.args.use_tqdm\
                else self.tqdm(enumerate(self.train_loader),
                            total=len(self.train_loader),
                            desc="training",
                            leave=True)

            for i, batch_data in train_iterator:
                loss = self._process_batch(batch_data)
                train_loss += loss.item()

                if self.args.use_tqdm:
                    train_iterator.set_postfix({"train_loss": float(loss),})

            train_loss = train_loss / len(self.train_loader)

            if self.args.use_tqdm:
                epochs.set_postfix({"Train Loss": train_loss,})
            else:
                print(f"EPOCH {epoch} | {train_loss}")

            self.early_stopping(train_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping.")
                break

        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.to(self.args.device)

        return self.model

    def _process_batch(self, batch_data):
        pass

class Tester:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        if self.args.use_tqdm:
            tqdms = {
                "plain": tqdm_plain,
                "notebook": tqdm_notebook,
                "gui": tqdm_gui,
            }
            self.tqdm = tqdms[args.tqdmopt]
            self.train_iterator = self.tqdm(enumerate(self.train_loader),
                                           total=len(self.train_loader),
                                           desc="train iterator",
                                           leave=True)
            self.test_iterator = self.tqdm(enumerate(self.test_loader),
                                            total=len(self.test_loader),
                                            desc="test iterator",
                                            leave=True)
        else:
            self.train_iterator = enumerate(self.train_loader)
            self.test_iterator = enumerate(self.test_loader)

    def get_anomaly_score(self):
        pass

    def plot_anomaly(self, anomaly_scores, intervals, threshold = None):
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot()

        # plot with horizontal threshold line
        if threshold:
            ax.axhline(y=threshold, color="r")

        x = list(range(len(anomaly_scores)))
        y = anomaly_scores
        ax.plot(x, y)
        plt.rcParams['axes.facecolor'] = 'white'
        for s, e in intervals:
            ax.axvspan(s, e, alpha=0.2, color='orange')

        plt.savefig(os.path.join(self.args.result_path, "threshold.png"))
        plt.show()

    def get_statistics(self, target_y, anomaly_prediction):
        '''
        :param target_y:
        :param anomaly_prediction:
        :return: accuracy, precision, recall, f1
        '''
        cm = confusion_matrix(target_y, anomaly_prediction)
        a, p, r, f = accuracy_score(target_y, anomaly_prediction),\
                  precision_score(target_y, anomaly_prediction, zero_division=1), \
                  recall_score(target_y, anomaly_prediction, zero_division=1), \
                  f1_score(target_y, anomaly_prediction, zero_division=1)
        return cm, a, p, r, f

    def regular_thresholding(self, test_y, anomaly_scores, intervals, num_candidates = 100):
        '''
        evenly devided threshold candidates between max_anomaly_score and min_anomaly_score
        :param anomaly_scores:
        :return: threshold with best F1-score
        '''

        # regular interval
        anomaly_min, anomaly_max = np.min(anomaly_scores), np.max(anomaly_scores)
        thresholds = np.linspace(anomaly_min, anomaly_max, num_candidates)

        # find best threshold
        print("finding best threshold...")
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for threshold in thresholds:
            anomaly_prediction = (anomaly_scores > threshold).astype(int)
            target_y = test_y[self.args.window_size - 1:]

            _, a, p, r, f = self.get_statistics(target_y, anomaly_prediction)
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            f1.append(f)
        print("done.")

        # metrics plot
        print("plotting metrics...")
        plt.plot(thresholds, accuracy, label="accuracy")
        plt.plot(thresholds, precision, label="precision")
        plt.plot(thresholds, recall, label="recall")
        plt.plot(thresholds, f1, label="f1")
        plt.legend()
        plt.savefig(os.path.join(self.args.result_path, "metrics.png"))
        plt.show()

        threshold_idx = np.argmax(f1)
        best_threshold = thresholds[threshold_idx]

        # threshold plot
        print("plotting anomaly scores with thresholds...")
        self.plot_anomaly(anomaly_scores, intervals, best_threshold)

        # print final statistics
        print("plotting final statistics...")
        anomaly_prediction = (anomaly_scores > best_threshold).astype(int)
        target_y = test_y[self.args.window_size - 1:]
        cm, a, p, r, f = self.get_statistics(target_y, anomaly_prediction)
        print(f"best threshold: {best_threshold}")
        print(f"accuracy: {a} "
              f"precision {p} "
              f"recall {r} "
              f"f1 {f}" )
        sns.heatmap(cm, annot = True, fmt = "d")

        return best_threshold