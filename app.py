import json
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import process_video

st.set_page_config(
    page_title="Bowling Scoreboard Extractor",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 Bowling Scoreboard Data Extraction")
st.write("Computer Vision + OCR pipeline for extracting scoreboard data from video.")

uploaded = st.file_uploader(
    "Upload bowling scoreboard video",
    type=["mp4", "mov", "avi"]
)

if uploaded:
    st.video(uploaded)

    if st.button("Extract Scoreboard", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / uploaded.name
            video_path.write_bytes(uploaded.getbuffer())

            with st.spinner("Detecting scoreboard and running OCR..."):
                result = process_video(
                    str(video_path),
                    "output/scoreboard.json"
                )

        st.success("Extraction completed.")

        st.subheader("Extracted data")
        st.json(result)

        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2),
            file_name="scoreboard.json",
            mime="application/json",
        )
