from torch.utils.data import DataLoader, Dataset
import numpy as np

class SWaTDataset(Dataset):
    def __init__(self, x, y, x_min, x_max, window_size=1):
        super().__init__()
        t = (x_min != x_max).astype(np.float32)
        self.x = (x - x_min) / (x_max-x_min + 1e-5) * t
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return self.x.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return self.x[idx:idx+self.window_size], self.y[idx:idx+self.window_size]