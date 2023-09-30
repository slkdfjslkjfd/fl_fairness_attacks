from torch.utils.data import Dataset

class ClassSubsetDataset(Dataset):

    def __init__(self, dataset, classes=[0, 1], num=int(1e10)):
        self.dataset = dataset
        self.indexes = [i for i, (__, y) in enumerate(self.dataset) if y in classes][:num]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]