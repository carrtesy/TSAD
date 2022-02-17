import torch
import torch.nn.functional as F
import numpy as np

from exp_helpers.exp import Trainer, Tester


class OmniAnomaly_Trainer(Trainer):
    def __init__(self, args, model, train_loader, loss_fn, optimizer, scheduler = None):
        super(OmniAnomaly_Trainer, self).__init__(args, model, train_loader, loss_fn, optimizer, scheduler)

    def _process_batch(self, batch_data):
        batch_data = batch_data[0].to(self.args.device)
        predict_values, mu, logvar = self.model(batch_data)

        self.optimizer.zero_grad()
        loss = self.loss_fn(batch_data, predict_values, mu, logvar)
        loss.backward()
        self.optimizer.step()

        return loss

class OmniAnomaly_Tester(Tester):
    def __init__(self, args, model, train_loader, test_loader):
        super(OmniAnomaly_Tester, self).__init__(args, model, train_loader, test_loader)
        loss_fn = F.mse_loss

        self.train_loss_list = self.get_loss_list(self.train_iterator, loss_fn)
        self.test_loss_list = self.get_loss_list(self.test_iterator, loss_fn)

    def get_anomaly_score(self):
        anomaly_scores = np.mean(self.test_loss_list, axis = 1)

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
                predict_values, _, _ = self.model(batch_data)

                loss = loss_fn(batch_data, predict_values, reduce=False)
                loss = loss.mean(dim=1).cpu().numpy()
                loss_list.append(loss)

        loss_list = np.concatenate(loss_list, axis=0)
        return loss_list
