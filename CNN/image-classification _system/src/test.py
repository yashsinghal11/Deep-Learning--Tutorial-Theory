import torch
from model import CNN
from load_data import test_loader

# Load trained model
model = CNN()

model.load_state_dict(
    torch.load("../data/cifar10_cnn.pth", map_location="cpu")
)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        # Select class with highest score
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Total test images:", total)
print("Correct predictions:", correct)
print(f"Test Accuracy: {accuracy:.2f}%")