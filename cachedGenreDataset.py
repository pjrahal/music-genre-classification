import numpy as np
import os
import torch

from collections import OrderedDict
from torch.utils.data import Dataset


class CachedGenreDataset(Dataset):
    def __init__(self, df_meta, features_dir, cache_size=500):
        self.df_meta = df_meta
        self.features_dir = features_dir
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        track_id = row.track_id
        label = row.label

        if track_id in self.cache:
            mel_db = self.cache[track_id]
        else:
            feature_path = os.path.join(self.features_dir, f"{track_id}.npy")
            mel_db = np.load(feature_path)
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # FIFO policy
            self.cache[track_id] = mel_db

        mel_db = mel_db[np.newaxis, :, :] if mel_db.ndim == 3 else mel_db
        return torch.tensor(mel_db, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
