from pathlib import Path
import cv2

from src.detector import is_scoreboard_frame, draw_detection

VIDEO = "input/bowling_scoreboard.mp4"
OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
saved = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if is_scoreboard_frame(frame) and saved < 3:
        cv2.imwrite(str(OUT / f"input_frame_{saved+1}.jpg"), frame)
        cv2.imwrite(str(OUT / f"detected_scoreboard_{saved+1}.jpg"),
                    draw_detection(frame))
        saved += 1

    if saved >= 3:
        break

cap.release()
print(f"Saved {saved} screenshot pairs in {OUT}/")
