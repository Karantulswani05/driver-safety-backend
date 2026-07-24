import time
import numpy as np
import cv2
from ultralytics import YOLO


# Load model once
model = YOLO("best.pt")

CLASS_MAP = {
    0: "Awake", 9: "Awake",
    2: "Sleeping", 5: "Sleeping",
    6: "Sleeping", 7: "Sleeping",
    1: "Drowsy",
    3: "Yawning", 4: "Yawning",
    10: "Yawning"
}

# Logic memory
sleep_start_time = 0
drowsy_eye_start = 0

is_sleep_active = False
is_drowsy_eye_active = False

yawn_timestamps = []
is_currently_yawning = False

CONF_THRESHOLD = 0.45
SLEEP_THRESHOLD_SEC = 2.0
DROWSY_THRESHOLD_SEC = 3.0
YAWN_LIMIT_PER_MIN = 5


def run_drowsiness_frame(frame):

    global sleep_start_time
    global drowsy_eye_start
    global is_sleep_active
    global is_drowsy_eye_active
    global yawn_timestamps
    global is_currently_yawning

    alert_message = "NORMAL"

    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    current_frame_labels = []

    for box in results[0].boxes:

        cls_id = int(box.cls)

        if cls_id in CLASS_MAP:

            label = CLASS_MAP[cls_id]
            current_frame_labels.append(label)

    now = time.time()

    is_yawning = "Yawning" in current_frame_labels
    is_sleeping = "Sleeping" in current_frame_labels
    is_drowsy_eye = "Drowsy" in current_frame_labels

    # --- Sleep Alert ---

    if is_sleeping:

        if not is_sleep_active:
            sleep_start_time = now
            is_sleep_active = True

        if (now - sleep_start_time) >= SLEEP_THRESHOLD_SEC:

            alert_message = "SLEEP ALERT"

    else:

        is_sleep_active = False

    # --- Drowsy Alert ---

    if is_drowsy_eye:

        if not is_drowsy_eye_active:
            drowsy_eye_start = now
            is_drowsy_eye_active = True

        if (now - drowsy_eye_start) >= DROWSY_THRESHOLD_SEC:

            alert_message = "DROWSY EYES"

    else:

        is_drowsy_eye_active = False

    # --- Yawn Alert ---

    if is_yawning:

        if not is_currently_yawning:

            yawn_timestamps.append(now)
            is_currently_yawning = True

    else:

        is_currently_yawning = False

    yawn_timestamps = [
        t for t in yawn_timestamps
        if now - t < 60
    ]

    if len(yawn_timestamps) >= YAWN_LIMIT_PER_MIN:

        alert_message = "TAKE COFFEE BREAK"

    return {
        "alert": alert_message
    }