import os
from dotenv import load_dotenv

load_dotenv()

# =============================
# MODEL
# =============================
MODEL_PATH = "models/y_hsv2.pt"

# =============================
# ALERT SETTINGS
# =============================
ALERT_THRESHOLD = 3
ALERT_COOLDOWN = 30

# =============================
# DISTANCE SETTINGS
# =============================
KNOWN_WIDTH = 2.5
FOCAL_LENGTH = 750
MAX_DISTANCE_CM = 45

# =============================
# TWILIO
# =============================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")