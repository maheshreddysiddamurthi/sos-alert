import time

class AlertDispatchAgent:
    def __init__(self, contact_list=None):
        if contact_list is None:
            contact_list = ["Family Member", "Police"]
        self.contacts = contact_list

    def send_alert(self, message):
        for contact in self.contacts:
            print(f"Alert sent to {contact}: {message}")
            print("Watch vibrating... Alert active.")
            time.sleep(1)
