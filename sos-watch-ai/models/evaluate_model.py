import torch
from torch.utils.data import DataLoader
from data_preprocessing import UrbanSoundDataset
from model import SoundClassifier

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SoundClassifier(num_classes=10).to(device)
model.load_state_dict(torch.load('sound_classifier.pth'))
model.eval()

# Prepare evaluation dataset
eval_dataset = UrbanSoundDataset('./data/UrbanSound8K', sr=22050, n_mels=128)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Evaluation Accuracy: {accuracy:.2f}%")
