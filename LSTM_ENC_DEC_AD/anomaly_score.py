import numpy as np
import torch
from tqdm.notebook import tqdm

class Anomaly_Calculator:
    def __init__(self, train_loss_list):
        self.train_loss_list = train_loss_list
        self.mean = np.mean(self.train_loss_list, axis=0)
        self.std = np.cov(self.train_loss_list.T)
        assert self.mean.shape[0] == self.std.shape[0] and self.mean.shape[0] == self.std.shape[1]

    def __call__(self, recons_error:np.array):
        x = (recons_error-self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)

    def get_anomaly_score_list(self, loss_list):
        anomaly_scores = []
        for item in tqdm(loss_list):
            score = self.__call__(item)
            anomaly_scores.append(score)

        print(f"=== Anomaly statistics ===\n"
              f"Total: {len(anomaly_scores)}\n"
              f"mean[{np.mean(anomaly_scores)}], median[{np.median(anomaly_scores)}], min[{np.min(anomaly_scores)}], max[{np.max(anomaly_scores)}]")
        return anomaly_scores