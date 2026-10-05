import json
import subprocess

STRUCTURE_FILE = "data/structure/test_blank_space.json"
INPUT_FILE = "data/songs/test_blank_space.wav"
OUTPUT_FILE = "data/choruses/test_blank_space_allin1.wav"

with open(STRUCTURE_FILE, "r") as f:
    data = json.load(f)

choruses = [
    segment
    for segment in data["segments"]
    if segment["label"] == "chorus"
]

if not choruses:
    raise ValueError("No chorus found")

first_chorus = choruses[0]

start = first_chorus["start"]

print("first chorus starts at:", start)

subprocess.run([
    "./tools/ffmpeg",
    "-y",
    "-ss", str(start),
    "-i", INPUT_FILE,
    "-t", "15",
    OUTPUT_FILE
], check=True)

print("saved to:", OUTPUT_FILE)