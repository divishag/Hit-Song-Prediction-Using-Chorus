import pandas as pd
import subprocess
import json
import os
import re
import shutil
from pathlib import Path

INPUT_CSV = "data/final_candidate_songs.csv"

SONGS_DIR = Path("data/songs")
CHORUS_DIR = Path("data/choruses")
STRUCTURE_DIR = Path("data/structure")

SONGS_DIR.mkdir(parents=True, exist_ok=True)
CHORUS_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

# START SMALL.
# Change this later after we confirm everything works.
LIMIT = 10


def safe_name(text):
    text = str(text)

    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())

    return text[:120]


df = pd.read_csv(INPUT_CSV)

# Only process first LIMIT songs for the first test
df = df.head(LIMIT)

success = []
failed = []


for index, row in df.iterrows():

    artist = row["artist"]
    song = row["song"]
    hit = row["hit"]

    base = safe_name(f"{artist}_{song}")

    mp3_file = SONGS_DIR / f"{base}.mp3"
    wav_file = SONGS_DIR / f"{base}.wav"
    chorus_file = CHORUS_DIR / f"{base}.wav"

    print("\n========================================")
    print(f"{index + 1}/{len(df)}")
    print(f"{artist} - {song}")
    print("========================================")

    # Skip songs already successfully processed
    if chorus_file.exists():
        print("chorus already exists - skipping")

        success.append({
            "artist": artist,
            "song": song,
            "hit": hit,
            "chorus_file": str(chorus_file)
        })

        continue

    try:

        # --------------------------------------------------
        # 1. DOWNLOAD AUDIO
        # --------------------------------------------------

        query = f"{artist} {song} official audio"

        print("downloading...")

        subprocess.run([
            "yt-dlp",
            f"ytsearch1:{query}",
            "-x",
            "--audio-format", "mp3",
            "--ffmpeg-location", "./tools",
            "-o", str(mp3_file)
        ], check=True)


        # --------------------------------------------------
        # 2. CONVERT MP3 -> WAV
        # --------------------------------------------------

        print("converting to wav...")

        subprocess.run([
            "./tools/ffmpeg",
            "-y",
            "-i", str(mp3_file),
            str(wav_file)
        ], check=True)


        # --------------------------------------------------
        # 3. RUN ALL-IN-ONE
        # --------------------------------------------------

        print("detecting song structure...")

        env = os.environ.copy()

        tools_path = str(Path("tools").resolve())

        env["PATH"] = tools_path + os.pathsep + env.get("PATH", "")

        subprocess.run([
            "allin1",
            str(wav_file),
            "--out-dir",
            str(STRUCTURE_DIR)
        ], check=True, env=env)


        # --------------------------------------------------
        # 4. READ ALL-IN-ONE OUTPUT
        # --------------------------------------------------

        structure_file = STRUCTURE_DIR / f"{wav_file.stem}.json"

        if not structure_file.exists():
            raise RuntimeError("All-In-One JSON output not found")

        with open(structure_file, "r") as f:
            structure = json.load(f)

        choruses = [
            segment
            for segment in structure["segments"]
            if segment["label"].lower() == "chorus"
        ]

        if not choruses:
            raise RuntimeError("No chorus detected")


        # --------------------------------------------------
        # 5. FIRST DETECTED CHORUS
        # --------------------------------------------------

        first_chorus = choruses[0]

        start = first_chorus["start"]
        end = first_chorus["end"]

        duration = end - start

        # Stanford uses 15 seconds
        extract_duration = min(15, duration)

        print(
            f"chorus detected: "
            f"{start:.2f}s - {end:.2f}s"
        )


        # --------------------------------------------------
        # 6. EXTRACT CHORUS
        # --------------------------------------------------

        subprocess.run([
            "./tools/ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(wav_file),
            "-t", str(extract_duration),
            str(chorus_file)
        ], check=True)


        success.append({
            "artist": artist,
            "song": song,
            "hit": hit,
            "chorus_start": start,
            "chorus_end": end,
            "chorus_file": str(chorus_file)
        })


        # --------------------------------------------------
        # 7. DELETE FULL SONG AUDIO
        # --------------------------------------------------

        if mp3_file.exists():
            mp3_file.unlink()

        if wav_file.exists():
            wav_file.unlink()

        print("SUCCESS")


    except Exception as e:

        print("FAILED:", e)

        failed.append({
            "artist": artist,
            "song": song,
            "hit": hit,
            "error": str(e)
        })


# --------------------------------------------------
# SAVE PROCESSING LOGS
# --------------------------------------------------

if success:
    pd.DataFrame(success).to_csv(
        "data/processed_songs.csv",
        index=False
    )

if failed:
    pd.DataFrame(failed).to_csv(
        "data/failed_songs.csv",
        index=False
    )


print("\n========================================")

print("successful:", len(success))
print("failed:", len(failed))

print("========================================")