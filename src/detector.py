import cv2
import numpy as np
from .config import ROI, DETECTION_V_LINES, DETECTION_H_LINES


def grid_features(frame):
    small = cv2.resize(frame, (480, 270))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    vertical_projection = (edges > 0).sum(axis=0)
    horizontal_projection = (edges > 0).sum(axis=1)

    v_lines = int((vertical_projection > 40).sum())
    h_lines = int((horizontal_projection > 50).sum())

    return v_lines, h_lines


def is_scoreboard_frame(frame):
    v_lines, h_lines = grid_features(frame)
    return v_lines >= DETECTION_V_LINES and h_lines >= DETECTION_H_LINES


def crop_scoreboard(frame):
    x1, y1, x2, y2 = ROI
    return frame[y1:y2, x1:x2].copy()


def draw_detection(frame):
    x1, y1, x2, y2 = ROI
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(
        out, "SCOREBOARD DETECTED",
        (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
    )
    return out
