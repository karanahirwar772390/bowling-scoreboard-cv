from src.pipeline import process_video

if __name__ == "__main__":
    result = process_video()
    print("\nProcessing complete.")
    print("Output: output/scoreboard.json")
    print(f"Detected scoreboard samples: {len(result['records'])}")
