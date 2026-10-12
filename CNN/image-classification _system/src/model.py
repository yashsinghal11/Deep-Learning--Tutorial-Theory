import torch
import torch.nn as nn


class CNN(nn.Module):
     def __init__(self):
          super().__init__()

          self.conv1 = nn.Conv2d(
               in_channels=3,
               out_channels=16,
               kernel_size=3,
               padding=1
          )

          self.relu = nn.ReLU()
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          self.conv2 = nn.Conv2d(
               in_channels=16,
               out_channels=32,
               kernel_size=3,
               padding=1
               )
          self.relu2 = nn.ReLU()
          self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
          self.flatten = nn.Flatten()
          self.fc1 = nn.Linear(in_features=2048, out_features=128)
          self.fc2 = nn.Linear(in_features=128, out_features=10)
     def forward(self, x):
          x = self.conv1(x)
          x = self.relu(x)
          x = self.pool(x)

          x = self.conv2(x)
          x = self.relu2(x)
          x = self.pool2(x)
          x = self.flatten(x)
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)

          return x
     
     
     
     