class RiskAssessAgent:
    def __init__(self):
        self.emergency_keywords = ["gun_shot", "siren", "scream",
"dog_bark"]
    def assess(self, label):
        if label in self.emergency_keywords:
            return True
        return False