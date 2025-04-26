import torch
import librosa
from model import SoundClassifier
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText

class SoundDetectAgent:
    def __init__(self, model_path='sound_classifier.pth', threshold=0.9):
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SoundClassifier(num_classes=10).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode
        self.threshold = threshold  # Threshold for triggering the alert
        print(f"SoundDetectAgent Initialized with model {model_path}")

    def process_audio(self, file_path, sr=22050, n_mels=128, duration=4):
        """
        Process the incoming audio file, extract features, and classify the sound.
        """
        # Load audio
        y, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Convert to Mel spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        # Convert to tensor and reshape
        inputs = torch.tensor(log_mels).unsqueeze(0).unsqueeze(0).to(self.device)

        # Classify using the model
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

        return predicted.item(), confidence

    def trigger_alert(self, sound_class, confidence):
        """
        Trigger an alert when a certain sound is detected with a high enough confidence.
        """
        alert_message = f"Alert: A sound of class '{sound_class}' was detected with {confidence*100:.2f}% confidence!"

        # For now, we will simulate the alert by sending an email
        self.send_email_alert(alert_message)

    def send_email_alert(self, message):
        """
        Send an email alert to the specified email address (family or police).
        """
        try:
            # Replace these with actual email credentials and recipient info
            sender_email = "smkr9933@gmail.com"
            receiver_email = "family_or_police@example.com"
            password = "your_email_password"

            msg = MIMEText(message)
            msg['Subject'] = "Emergency Alert: Sound Detected"
            msg['From'] = sender_email
            msg['To'] = receiver_email

            # Connect to the server and send the email
            with smtplib.SMTP('smtp.example.com', 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())

            print("Alert email sent!")

        except Exception as e:
            print(f"Failed to send email: {e}")

    def monitor_sound(self, audio_file_path):
        """
        Continuously monitor sound and trigger alerts if specific sounds are detected.
        """
        while True:
            # Process the audio and get predictions
            sound_class, confidence = self.process_audio(audio_file_path)

            # Check if the confidence is above the threshold
            if confidence >= self.threshold:
                self.trigger_alert(sound_class, confidence)

            # Wait before processing the next audio file (you can adjust the interval as needed)
            time.sleep(5)  # You can change this to process in real time with a live stream of audio
