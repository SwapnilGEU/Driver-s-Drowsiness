import time
from config.settings import ALERT_COOLDOWN

last_alert_time = 0


def can_send_alert():

    global last_alert_time

    current_time = time.time()

    if current_time - last_alert_time >= ALERT_COOLDOWN:
        last_alert_time = current_time
        return True

    return False