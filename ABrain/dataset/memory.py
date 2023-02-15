from torch.utils.data import DataLoader, Dataset, IterDataPipe
from torchio.transforms import Transform


class MemorySet(Dataset):
    def __init__(self, dataset: Dataset, transforms: Transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.table = list(None for _ in range(len(dataset)))
        self.trans = transforms

    def _get_slow(self, index):
        data = self.dataset[index]
        if self.trans:
            data = self.trans(data)
        self.table[index] = data
        return data

    def __getitem__(self, index):
        if self.table[index]:
            return self.table[index]
        else:
            return self._get_slow(index)

    def __len__(self):
        return len(self.dataset)
