import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class UrbanSoundDataset(Dataset):
    def __init__(self, data_dir, sr=22050, n_mels=128, duration=4):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.classes = os.listdir(data_dir)
        self.data = self.load_data()

    def load_data(self):
        data = []
        labels = []
        for class_id, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_folder):  # Skip files, process only directories
                continue
            for file_name in os.listdir(class_folder):
                if file_name.endswith(".wav"):  # Only process .wav files
                    file_path = os.path.join(class_folder, file_name)
                    data.append(file_path)
                    labels.append(class_id)

        return list(zip(data, labels))

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        return torch.tensor(log_mels).unsqueeze(0), label

    def __len__(self):
        return len(self.data)  # Return the length of the dataset
