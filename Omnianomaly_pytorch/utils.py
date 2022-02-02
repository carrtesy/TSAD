import numpy as np
import torch
from tqdm.notebook import tqdm

def get_loss_list(args, model, loader, loss_fn):
    iterator = tqdm(enumerate(loader), total=len(loader), desc="Getting Loss List")
    loss_list = []

    model.eval()
    with torch.no_grad():
        for i, batch_data in iterator:
            batch_data = batch_data[0].to(args.device)
            predict_values, _, _ = model(batch_data)

            loss = loss_fn(batch_data, predict_values, reduce=False)
            loss = loss.mean(dim=1).cpu().numpy()
            loss_list.append(loss)

    loss_list = np.concatenate(loss_list, axis=0)
    print(f"loss list created: {loss_list.shape}")
    return loss_list