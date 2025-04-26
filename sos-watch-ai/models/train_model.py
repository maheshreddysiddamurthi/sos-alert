import torch  # <-- Make sure torch is imported
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from data_preprocessing import UrbanSoundDataset
from model import SoundClassifier

# Setup
data_dir = '/Users/office/Documents/sos/sos-alert/sos-watch-ai/.data/UrbanSound8K'  # Corrected path
batch_size = 32
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset and dataloaders
train_dataset = UrbanSoundDataset(data_dir, sr=22050, n_mels=128)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup
model = SoundClassifier(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear the gradients
        outputs = model(inputs)  # Forward pass

        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters

        total_loss += loss.item()  # Track the loss
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total += labels.size(0)  # Update total number of samples
        correct += (predicted == labels).sum().item()  # Update number of correct predictions

    # Print stats for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")
