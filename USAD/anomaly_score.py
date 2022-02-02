import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np

class Anomaly_Calculator:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def get_anomaly_score(self, Wt, model, alpha=0.5, beta=0.5):
        z = model.encoder(Wt)
        Wt1p = model.decoder1(z)
        Wt2dp = model.decoder2(model.encoder(Wt1p))
        return alpha * F.mse_loss(Wt, Wt1p, reduce = False) + beta * F.mse_loss(Wt, Wt2dp, reduce = False)

    def get_anomaly_score_list(self, loader, alpha = 0.5, beta = 0.5):
        iterator = tqdm(enumerate(loader), total=len(loader), desc="Getting anomaly score")
        anomaly_scores = []
        with torch.no_grad():
            for i, batch_data in iterator:
                batch_data = batch_data[0].to(self.args.device)
                anomaly_score = self.get_anomaly_score(batch_data, self.model, alpha, beta).mean(axis=(1, 2)).to("cpu")
                anomaly_scores.append(anomaly_score)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)

        print(f"=== Anomaly statistics ===\n"
              f"Total: {len(anomaly_scores)}\n"
              f"mean[{np.mean(anomaly_scores)}], median[{np.median(anomaly_scores)}], min[{np.min(anomaly_scores)}], max[{np.max(anomaly_scores)}]")
        return anomaly_scores