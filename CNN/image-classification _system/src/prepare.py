from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="../data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
print("Total training images:", len(train_dataset))
print("Total batches:", len(train_loader))

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)