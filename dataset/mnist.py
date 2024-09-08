from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Callable, Optional

DATA_TRANSFORM = {
    'train': transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
        ]
    )
}


class MnistDataset(Dataset):

    def __init__(self, image_dir: str, transformer: Optional[Callable] = None):
        image_suffix = ['.png', '.jpg']
        self.transformer = transformer
        self.images_path = [f for f in Path(image_dir).iterdir() if f.suffix in image_suffix]

    def __getitem__(self, index):
        image = self.images_path[index]
        if self.transformer:
            image = self.transformer(image)

        return image

    def __len__(self):
        return len(self.images_path)
