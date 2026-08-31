# Computer Vision Engineer Assessment
## Bowling Scoreboard Data Extraction

### Candidate
Karan Ahirwar

## 1. Objective
Automatically detect and extract scoreboard information from the supplied bowling video.

## 2. System Architecture

Video
-> Frame Sampling
-> Scoreboard Detection
-> Scoreboard ROI
-> Image Preprocessing
-> OCR
-> Post-processing
-> Temporal Stabilization
-> JSON

## 3. Input Video
Insert screenshot: `screenshots/input_frame_1.jpg`

Brief explanation:
The system receives the supplied MP4 video as input.

## 4. Scoreboard Detection
Insert screenshot: `screenshots/detected_scoreboard_1.jpg`

Brief explanation:
The detector identifies the scoreboard using its strong grid structure and isolates the scoreboard ROI.

## 5. OCR
Insert screenshot of the application while processing.

Brief explanation:
Multiple image channels are evaluated before Tesseract OCR to improve recognition on colored scoreboard backgrounds.

## 6. Structured Output
Insert screenshot of `output/scoreboard.json` or Streamlit output.

Example:

```json
{
  "active_player": "VISHAL",
  "totals": {
    "J": 41,
    "V": 37,
    "P": 54,
    "T": 40
  }
}
```

## 7. Temporal Consistency
A short history of OCR predictions is maintained and majority voting is used to reduce one-frame recognition errors.

## 8. Result
The final system automatically detects scoreboard frames and produces machine-readable scoreboard data.
