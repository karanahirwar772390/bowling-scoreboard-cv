import re
from difflib import SequenceMatcher

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

from .config import NAME_ROI, TOTAL_CENTERS


# ---------------------------------------------------------
# Known players from the scoreboard
# ---------------------------------------------------------

KNOWN_PLAYERS = [
    "TARUN",
    "JAGDISH",
    "VISHAL",
]


# ---------------------------------------------------------
# OCR text normalization
# ---------------------------------------------------------

def normalize_ocr_text(text):
    replacements = {
        "|": "1",
        "I": "1",
        "l": "1",
        "]": "1",
        "O": "0",
        "o": "0",
        "S": "5",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    return text


# ---------------------------------------------------------
# Clean player name
# ---------------------------------------------------------

def clean_name(text):
    text = text.upper()

    # Remove common OCR junk around names
    text = re.sub(r"[^A-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Match noisy OCR name to known player
# ---------------------------------------------------------

def match_player_name(text):

    text = clean_name(text)

    if not text:
        return None

    # Exact match
    if text in KNOWN_PLAYERS:
        return text

    # Remove spaces for comparison
    compact = text.replace(" ", "")

    best_name = None
    best_score = 0.0

    for player in KNOWN_PLAYERS:

        player_compact = player.replace(" ", "")

        score = SequenceMatcher(
            None,
            compact,
            player_compact
        ).ratio()

        # Also check whether the real name appears inside
        # the noisy OCR result.
        if player_compact in compact:
            score = max(score, 0.90)

        if score > best_score:
            best_score = score
            best_name = player

    # Conservative threshold
    if best_score >= 0.60:
        return best_name

    return None


# ---------------------------------------------------------
# Extract active player
# ---------------------------------------------------------

def extract_active_name(frame):

    x1, y1, x2, y2 = NAME_ROI

    crop = frame[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    candidates = []

    # Try multiple OCR modes
    for psm in (7, 8, 13):

        text = pytesseract.image_to_string(
            crop,
            config=f"--psm {psm}"
        )

        name = match_player_name(text)

        if name:
            candidates.append(name)

    if not candidates:
        return None

    # Most common recognized player
    counts = {}

    for name in candidates:
        counts[name] = counts.get(name, 0) + 1

    return max(
        counts,
        key=counts.get
    )


# ---------------------------------------------------------
# OCR scoreboard channels
# ---------------------------------------------------------

def ocr_scoreboard_channels(frame):

    roi = frame[0:850, 0:1900]

    channels = {
        "gray": cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        ),
        "R": roi[:, :, 2],
        "G": roi[:, :, 1],
    }

    all_items = []

    for channel_name, image in channels.items():

        image = cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        data = pytesseract.image_to_data(
            image,
            config="--psm 6",
            output_type=pytesseract.Output.DICT
        )

        for i, raw in enumerate(data["text"]):

            raw = raw.strip()

            if not raw:
                continue

            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = 0

            # Ignore very low-confidence OCR
            if conf < 15:
                continue

            x = data["left"][i] / 1.5
            y = data["top"][i] / 1.5

            all_items.append({
                "text": normalize_ocr_text(raw),
                "confidence": conf,
                "x": x,
                "y": y,
                "channel": channel_name,
            })

    return all_items


# ---------------------------------------------------------
# Extract scoreboard totals
# ---------------------------------------------------------

def extract_totals_from_items(items):

    totals = {}

    for player, center_y in TOTAL_CENTERS.items():

        candidates = []

        for item in items:

            # Total values are on right side
            if item["x"] < 1700:
                continue

            # Match row
            if abs(item["y"] - center_y) > 55:
                continue

            value_text = re.sub(
                r"[^0-9]",
                "",
                item["text"]
            )

            if not value_text:
                continue

            if len(value_text) > 3:
                continue

            try:
                value = int(value_text)
            except ValueError:
                continue

            # Reject obviously impossible OCR values
            if value < 0 or value > 300:
                continue

            candidates.append(
                (
                    item["confidence"],
                    value,
                    item["text"]
                )
            )

        if candidates:

            # Highest confidence first
            candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            totals[player] = candidates[0][1]

        else:

            totals[player] = None

    return totals


# ---------------------------------------------------------
# Main OCR function
# ---------------------------------------------------------

def extract_scoreboard_data(frame):

    items = ocr_scoreboard_channels(frame)

    return {
        "active_player": extract_active_name(frame),
        "totals": extract_totals_from_items(items),
    }