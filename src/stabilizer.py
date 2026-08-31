from collections import Counter, deque


class TemporalStabilizer:
    def __init__(self, history_size=7):
        self.history = deque(maxlen=history_size)
        self.last_valid = None

    def update(self, value):
        # OCR value missing
        if value is None:
            return self.last_valid

        # First valid value
        if self.last_valid is None:
            self.last_valid = value
            self.history.append(value)
            return value

        # Reject obvious OCR truncation:
        # 54 -> 5, 41 -> 4, 31 -> 3
        if self.last_valid >= 10 and value < 10:
            return self.last_valid

        # Reject large OCR jumps
        if abs(value - self.last_valid) > 20:
            return self.last_valid

        self.history.append(value)

        # If current OCR value repeatedly appears,
        # accept it as the new stable score.
        counts = Counter(self.history)
        candidate, count = counts.most_common(1)[0]

        if candidate != self.last_valid and count >= 2:
            self.last_valid = candidate

        return self.last_valid