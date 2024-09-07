from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional


class MnistDataset(Dataset):

    def __init__(self, image_dir: str, transformer: Optional[Callable] = None):
        image_suffix = ['.png', '.jpg']
        self.transformer = transformer
        self.images_path = [f for f in Path(image_dir).iterdir() if f.suffix in image_suffix]

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.images_path)
