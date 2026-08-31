VIDEO_PATH = "input/bowling_scoreboard.mp4"

# Scoreboard occupies almost the full frame in this assessment video.
ROI = (20, 15, 1895, 1035)  # x1, y1, x2, y2

# Approximate vertical centres of the four Total-score cells.
TOTAL_CENTERS = {
    "J": 250,
    "V": 410,
    "P": 542,
    "T": 750,
}

# Top title containing the currently active player's name.
NAME_ROI = (220, 10, 950, 90)

FRAME_SAMPLE_STEP = 15       # OCR every 0.5 s for a 30 FPS video
DETECTION_V_LINES = 70
DETECTION_H_LINES = 80
