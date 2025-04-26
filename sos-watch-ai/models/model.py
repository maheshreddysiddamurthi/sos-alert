import torch
import torch.nn as nn

class SoundClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SoundClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function and pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        # Flatten the tensor before passing it to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
