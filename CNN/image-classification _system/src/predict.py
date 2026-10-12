
import torch
import matplotlib.pyplot as plt

from model import CNN
from load_data import test_dataset
import random
# CIFAR-10 classes
classes = test_dataset.classes

# Load trained model
model = CNN()
model.load_state_dict(
    torch.load("../data/cifar10_cnn.pth", map_location="cpu")
)
model.eval()

# Select one image


index = random.randrange(len(test_dataset))
image, actual_label = test_dataset[index]

# Predict
with torch.no_grad():
    image_batch = image.unsqueeze(0)
    outputs = model(image_batch)
    predicted_label = outputs.argmax(dim=1).item()

# Display image and result
plt.imshow(image.permute(1, 2, 0))
plt.title(
    f"Actual: {classes[actual_label]} | "
    f"Predicted: {classes[predicted_label]}"
)
plt.axis("off")
plt.show()

print("Actual class:", classes[actual_label])
print("Predicted class:", classes[predicted_label])
