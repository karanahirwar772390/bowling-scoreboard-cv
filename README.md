# Bowling Scoreboard Data Extraction from Video

## Overview

This project extracts bowling scoreboard information from a video using Computer Vision and OCR techniques.

The pipeline:
1. Reads the input video using OpenCV.
2. Detects/crops the scoreboard region.
3. Preprocesses the scoreboard image.
4. Extracts scoreboard information.
5. Generates structured JSON output.

## Technologies Used

- Python
- OpenCV
- NumPy
- OCR
- JSON

## Project Structure

```text
bowling-scoreboard-cv/
├── extract_scoreboard.py
├── requirements.txt
├── scoreboard_data.json
├── scoreboard_data_verified.json
├── detected_scoreboard.jpg
├── final_output_panel.png
├── scoreboard_extraction_report.pdf
├── demo_scoreboard.mp4
└── README.md
