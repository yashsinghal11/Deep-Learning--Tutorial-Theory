import matplotlib.pyplot as plt
from torchvision import datasets, transforms
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="../data",
    train=True,
    download=False,
    transform=transform
)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    image, label = train_dataset[i]
    axes[i].imshow(image.permute(1, 2, 0), interpolation="nearest")
    axes[i].set_title(classes[label])

plt.tight_layout()
plt.show()