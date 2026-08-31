import json
from pathlib import Path
import cv2

from .config import VIDEO_PATH, FRAME_SAMPLE_STEP
from .detector import is_scoreboard_frame
from .ocr_engine import extract_scoreboard_data
from .stabilizer import TemporalStabilizer


def process_video(video_path=VIDEO_PATH, output_json="output/scoreboard.json"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stabilizers = {
        "J": TemporalStabilizer(),
        "V": TemporalStabilizer(),
        "P": TemporalStabilizer(),
        "T": TemporalStabilizer(),
    }

    records = []
    last_name = None
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % FRAME_SAMPLE_STEP == 0 and is_scoreboard_frame(frame):
            data = extract_scoreboard_data(frame)

            if data["active_player"]:
                last_name = data["active_player"]

            stable_totals = {}

            for player, value in data["totals"].items():
                stable_totals[player] = stabilizers[player].update(value)

            records.append({
                "timestamp_sec": round(frame_index / fps, 2),
                "frame": frame_index,
                "active_player": last_name,
                "totals": stable_totals,
            })

        frame_index += 1

    cap.release()

    result = {
        "video": Path(video_path).name,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": round(frame_count / fps, 2),
        "records": records,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result