import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from load_data import train_loader


# 1. Initialize the CNN model
model = CNN()

# 2. Define the loss function
criterion = nn.CrossEntropyLoss()

# 3. Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Set the number of epochs
num_epochs = 5

# 5. Training loop
for epoch in range(num_epochs):

    total_loss = 0.0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss_value = criterion(outputs, labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss_value.backward()

        # Update model weights
        optimizer.step()

        # Accumulate batch loss
        total_loss += loss_value.item()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)

    # Display training progress
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )

print("Training completed!")
torch.save(model.state_dict(), "../data/cifar10_cnn.pth")
print("Model saved successfully!")