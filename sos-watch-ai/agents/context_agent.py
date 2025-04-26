import sounddevice as sd
import scipy.io.wavfile as wav
import os

class ContextAgent:
    def __init__(self, duration=5, out_dir="context_audio"):
        self.duration = duration
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def record_context(self, filename="context.wav"):
        fs = 44100
        print("Recording surroundings...")
        recording = sd.rec(int(self.duration * fs), samplerate=fs, channels=1)
        sd.wait()
        filepath = os.path.join(self.out_dir, filename)
        wav.write(filepath, fs, recording)
        print(f"Saved context audio: {filepath}")
        return filepath
