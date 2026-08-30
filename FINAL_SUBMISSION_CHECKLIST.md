# FOG Question 1 — Submission Checklist

## 1. GitHub
- [ ] Create a public/private GitHub repository.
- [ ] Upload `src/`, `requirements.txt`, `README.md`, `data/`, `output/`, and `docs/`.
- [ ] Do not upload unnecessary large temporary files.
- [ ] If the MP4 is too large for GitHub, keep it in Google Drive and mention the path in README.

## 2. Demo video
The provided `demo_scoreboard.mp4` demonstrates:
1. Input video
2. Scoreboard detection
3. Extraction stage
4. Final structured scoreboard output

For the strongest submission, record your screen while actually running:
```bash
python src/extract_scoreboard.py --video data/bowling_scoreboard.mp4 --output output
```
and show the terminal plus the generated `detected_scoreboard.jpg` and JSON.

## 3. PDF documentation
`docs/scoreboard_extraction_report.pdf` contains screenshots and short explanations for:
- Input frame
- Detected scoreboard ROI
- Extracted JSON/output
- Method and limitations

## 4. Final answer to the assessment
Mention:
- OpenCV for frame processing and scoreboard localization
- HSV/edge preprocessing for the scoreboard
- OCR/template matching for score symbols
- Temporal voting/validation for robustness
- JSON output as the final structured result
