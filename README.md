# Bowling Scoreboard Data Extraction

## 1. Problem

Extract scoreboard information automatically from `bowling_scoreboard.mp4`.

## 2. Approach

The solution uses:

1. OpenCV for video/frame processing.
2. A computer-vision scoreboard detector based on the distinctive grid structure.
3. ROI extraction.
4. Multi-channel image preprocessing.
5. Tesseract OCR for text/numeric recognition.
6. Temporal majority voting to reduce frame-level OCR errors.
7. JSON output for structured scoreboard data.
8. Streamlit for an easy demonstration UI.

### Pipeline

Video -> Frame Sampling -> Scoreboard Detection -> ROI -> Preprocessing -> OCR -> Temporal Stabilization -> JSON

## 3. Why not OCR every frame?

The supplied video is 30 FPS. OCR on every frame is unnecessary and expensive. The pipeline samples every 15 frames (~2 samples/sec) and uses temporal stabilization.

## 4. Setup

Install Python 3.10+ and Tesseract OCR.

### Windows Tesseract

Install Tesseract OCR and make sure `tesseract.exe` is in PATH.

If it is not in PATH, add this before importing/using pytesseract:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Then:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put the assessment video here:

```text
input/bowling_scoreboard.mp4
```

## 5. Run the pipeline

```bash
python run.py
```

Output:

```text
output/scoreboard.json
```

## 6. Run the demo

```bash
streamlit run app.py
```

Upload the video and click **Extract Scoreboard**.

## 7. Screenshots

Run:

```bash
python extract_screenshots.py
```

The script creates example input/detection screenshots under `screenshots/`.

## 8. Engineering decisions

### Scoreboard detection

The scoreboard contains many long vertical/horizontal grid lines. A small grayscale representation is processed with Canny edge detection and line-density projections. This separates scoreboard frames from bowling animation/cutaway frames without requiring model training.

### OCR

Different color channels are tested because the scoreboard contains blue, cyan, yellow and red backgrounds. Numeric OCR is performed on multiple channels and the highest-confidence result is selected.

### Temporal stabilization

OCR occasionally misreads a digit in a single frame. A short history and majority voting reduce these transient errors.

## 9. Limitations

- The supplied scoreboard has a stable camera/layout, so ROI coordinates are configured for this video.
- OCR accuracy depends on image quality and scoreboard overlays.
- For a different broadcast/layout, a learned detector such as YOLO could replace the grid-based detector.
