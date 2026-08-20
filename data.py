"""
Fashion-MNIST dataset loading, transforms, and DataLoaders.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Fashion-MNIST's known per-channel mean/std for normalization
MEAN, STD = (0.2860,), (0.3530,)


def get_transforms():
    """Return (train_transform, test_transform)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),   # mild augmentation: shifts, not flips
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, test_transform


def get_dataloaders(data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2):
    """Download (if needed) Fashion-MNIST and return (train_loader, test_loader)."""
    train_transform, test_transform = get_transforms()

    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when run directly: python data.py
    train_loader, test_loader = get_dataloaders()
    xb, yb = next(iter(train_loader))
    print(f"Batch shape: {xb.shape}")   # expect (128, 1, 28, 28)
    print(f"Labels shape: {yb.shape}")  # expect (128,)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    print(f"Classes: {CLASS_NAMES}")