import cv2
import time
import threading
import torch

from ultralytics import YOLO

from config.settings import (
    MODEL_PATH,
    ALERT_THRESHOLD,
    KNOWN_WIDTH,
    FOCAL_LENGTH,
    MAX_DISTANCE_CM
)

from detection.metrics import (
    calculate_fps,
    calculate_distance
)

from detection.overlays import (
    draw_status,
    draw_fps,
    draw_distance,
    draw_awake_bar
)

from alerts.sound_alert import play_alarm
from alerts.alert_manager import can_send_alert
from alerts.twilio_alert import send_emergency_alert


class DrowsinessDetector:

    def __init__(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(MODEL_PATH).to(device)

        self.drowsy_detected_time = None
        self.alert_played = False
        self.awake_percentage = 100

    def process_frame(
        self,
        frame,
        phone_number,
        voice_message,
        sms_message
    ):

        results = self.model(frame)[0]

        class_indices = results.boxes.cls
        class_names = [
            self.model.names[int(cls)]
            for cls in class_indices
        ]

        min_distance_cm = 100

        for box, cls_idx in zip(results.boxes.xyxy, class_indices):

            class_name = self.model.names[int(cls_idx)]

            if class_name == 'Closed Eye':

                x1, y1, x2, y2 = map(int, box)

                pixel_width = x2 - x1

                distance_cm = calculate_distance(
                    pixel_width,
                    KNOWN_WIDTH,
                    FOCAL_LENGTH
                )

                min_distance_cm = min(
                    min_distance_cm,
                    distance_cm
                )

        is_drowsy = (
            class_names.count('Closed Eye') >= 2
            and min_distance_cm < MAX_DISTANCE_CM
        )

        if is_drowsy:

            if self.drowsy_detected_time is None:
                self.drowsy_detected_time = time.time()

            elapsed = time.time() - self.drowsy_detected_time

            self.awake_percentage = max(
                0,
                100 - int((elapsed / ALERT_THRESHOLD) * 100)
            )

            if elapsed >= ALERT_THRESHOLD:

                if not self.alert_played:

                    threading.Thread(
                        target=play_alarm,
                        daemon=True
                    ).start()

                    self.alert_played = True

                if can_send_alert():

                    threading.Thread(
                        target=send_emergency_alert,
                        args=(
                            phone_number,
                            voice_message,
                            sms_message
                        ),
                        daemon=True
                    ).start()

        else:

            self.drowsy_detected_time = None
            self.alert_played = False
            self.awake_percentage = 100

        annotated_frame = results.plot()

        fps = calculate_fps()

        draw_status(annotated_frame, is_drowsy)
        draw_fps(annotated_frame, fps)
        draw_distance(annotated_frame, min_distance_cm)
        draw_awake_bar(annotated_frame, self.awake_percentage)

        return annotated_frame, is_drowsy