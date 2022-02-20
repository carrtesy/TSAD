import torch
import torch.nn.functional as F
import numpy as np

from exp_helpers.exp import Trainer, Tester

class USAD_Trainer(Trainer):
    def __init__(self, args, model, train_loader, loss_fn, optimizer, scheduler = None):
        super(USAD_Trainer, self).__init__(args, model, train_loader, loss_fn, optimizer, scheduler)

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
                loss = self._process_batch(batch_data, epoch+1)
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

    def _process_batch(self, batch_data, epoch):
        batch_data = batch_data[0].to(self.args.device)

        z = self.model.encoder(batch_data)
        Wt1p = self.model.decoder1(z)
        Wt2p = self.model.decoder2(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))

        self.optimizer.zero_grad()
        loss = self.loss_fn(batch_data, Wt1p, Wt2p, Wt2dp, epoch)
        loss.backward()
        self.optimizer.step()

        return loss


class USAD_Tester(Tester):
    def __init__(self, args, model, train_loader, test_loader):
        super(USAD_Tester, self).__init__(args, model, train_loader, test_loader)


    def anomaly_score_fn(self, Wt, alpha=0.5, beta=0.5):
        '''
        :param Wt: model input
        :param alpha: low detection sensitivity
        :param beta: high detection sensitivity
        :return: anomaly score
        '''
        z = self.model.encoder(Wt)
        Wt1p = self.model.decoder1(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))
        return alpha * F.mse_loss(Wt, Wt1p, reduce = False) + beta * F.mse_loss(Wt, Wt2dp, reduce = False)

    def get_anomaly_score(self, alpha = 0.5, beta = 0.5):
        self.model.eval()
        anomaly_scores = []
        with torch.no_grad():
            for i, batch_data in self.test_iterator:
                batch_data = batch_data[0].to(self.args.device)
                anomaly_score = self.anomaly_score_fn(batch_data, alpha, beta).mean(axis=(1, 2)).to("cpu")
                anomaly_scores.append(anomaly_score)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)

        print(f"=== Anomaly statistics ===\n"
              f"Total: {len(anomaly_scores)}\n"
              f"mean[{np.mean(anomaly_scores)}], "
              f"median[{np.median(anomaly_scores)}], "
              f"min[{np.min(anomaly_scores)}], "
              f"max[{np.max(anomaly_scores)}]")
        return anomaly_scores