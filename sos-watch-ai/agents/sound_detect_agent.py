import torchaudio
import librosa
import numpy as np
import torch
import os

class SoundDetectAgent:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.labels = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=22050)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mels = librosa.power_to_db(mels, ref=np.max)
        return torch.tensor(log_mels).unsqueeze(0).unsqueeze(0)

    def detect(self, audio_file):
        features = self.extract_features(audio_file)
        with torch.no_grad():
            prediction = self.model(features)
            label_idx = torch.argmax(prediction).item()
            return self.labels[label_idx]
