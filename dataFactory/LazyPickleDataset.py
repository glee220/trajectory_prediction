import dill
from torch.utils.data import Dataset

class LazyPickleDataset(Dataset):
    def __init__(self, pkl_path, index_path):
        self.pkl_path = pkl_path
        with open(index_path, 'r') as f:
            self.offsets = [int(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        with open(self.pkl_path, 'rb') as f:
            f.seek(offset)
            sample = dill.load(f)
        return sample
