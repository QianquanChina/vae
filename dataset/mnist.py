from torch.utils.data import Dataset
from typing import  Callable, Optional


class MnistDataset(Dataset):

    def __init__(self, image_dir, transformer: Optional[Callable] = None):
        self.image_dir = image_dir

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
