import cv2

from src.config import VIDEO_PATH, FRAME_SAMPLE_STEP
from src.detector import is_scoreboard_frame
from src.ocr_engine import extract_scoreboard_data
from src.stabilizer import TemporalStabilizer

cap = cv2.VideoCapture(VIDEO_PATH)

stabilizer = TemporalStabilizer()

frame_index = 0
count = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if frame_index % FRAME_SAMPLE_STEP == 0 and is_scoreboard_frame(frame):
        data = extract_scoreboard_data(frame)

        raw_p = data["totals"]["P"]
        stable_p = stabilizer.update(raw_p)

        print(
            f"{frame_index:4d} | "
            f"raw P={raw_p!s:>3} | "
            f"stable P={stable_p!s:>3}"
        )

        count += 1

        if count >= 15:
            break

    frame_index += 1

cap.release()