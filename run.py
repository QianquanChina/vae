from torch.utils.data import DataLoader
from dataset import DATA_TRANSFORM, MnistDataset

from models import VanillaVae


def main():
    # 数据处理
    train_dataset = MnistDataset(image_dir='', transformer=DATA_TRANSFORM['train'])
    val_dataset = MnistDataset(image_dir='', transformer=DATA_TRANSFORM['val'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # 模型加载
    model = VanillaVae(in_channels=3, latent_dim=128)


if __name__ == '__main__':
    main()
