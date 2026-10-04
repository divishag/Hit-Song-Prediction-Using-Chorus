import pandas as pd
import subprocess
import json
import os
import re
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

INPUT_CSV = "data/initial_550_songs.csv"
LOG_FILE = Path("data/download_extract_log.csv")

SONGS_DIR = Path("data/songs")
CHORUS_DIR = Path("data/choruses")
STRUCTURE_DIR = Path("data/structure")

SONGS_DIR.mkdir(parents=True, exist_ok=True)
CHORUS_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# SAFE FILE NAME
# ==========================================================

def safe_name(text):
    text = str(text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:120]


# ==========================================================
# NORMALIZE TEXT FOR YOUTUBE MATCHING
# ==========================================================

def normalize_text(text):
    text = str(text).lower()

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("‐", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# FIND BEST YOUTUBE RESULT
# ==========================================================

def get_youtube_info(artist, song):

    # Quoting the song title helps ambiguous songs like "Piano"
    query = f'{artist} "{song}" official audio'

    result = subprocess.run(
        [
            "yt-dlp",
            f"ytsearch5:{query}",
            "--dump-single-json",
            "--skip-download"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    data = json.loads(result.stdout)

    artist_norm = normalize_text(artist)
    song_norm = normalize_text(song)

    bad_words = [
        "karaoke",
        "cover",
        "reaction",
        "tutorial",
        "instrumental",
        "piano version",
        "slowed",
        "reverb",
        "nightcore",
        "sped up",
        "1 hour",
        "one hour",
        "30 minutes",
        "playlist",
        "best hits",
        "greatest hits",
        "full album",
        "compilation",
        "medley"
    ]

    candidates = []

    for entry in data.get("entries", []):

        if not entry:
            continue

        title = entry.get("title", "")
        uploader = entry.get("uploader", "")
        channel = entry.get("channel", "")
        duration = entry.get("duration")

        title_norm = normalize_text(title)
        uploader_norm = normalize_text(uploader)
        channel_norm = normalize_text(channel)

        # --------------------------------------------------
        # REJECT OBVIOUSLY BAD RESULTS
        # --------------------------------------------------

        if any(
            bad_word in title_norm
            for bad_word in bad_words
        ):
            continue

        # Normal songs should not be ridiculously long
        if duration is not None:

            if duration < 60:
                continue

            if duration > 600:
                continue

        # --------------------------------------------------
        # SCORE RESULT
        # --------------------------------------------------

        score = 0

        # Exact song title appearing in YouTube title
        if song_norm in title_norm:
            score += 10

        # Artist in title
        if artist_norm in title_norm:
            score += 8

        # Artist in uploader/channel
        if artist_norm in uploader_norm:
            score += 6

        if artist_norm in channel_norm:
            score += 6

        # Official audio/video strongly preferred
        if "official audio" in title_norm:
            score += 6

        if "official video" in title_norm:
            score += 5

        if "official music video" in title_norm:
            score += 5

        # Artist Topic channels are often official releases
        if "topic" in uploader_norm:
            score += 4

        if "topic" in channel_norm:
            score += 4

        if "vevo" in uploader_norm:
            score += 4

        if "vevo" in channel_norm:
            score += 4

        # Lyrics videos are acceptable but less preferred
        if "lyrics" in title_norm:
            score -= 1

        candidates.append(
            {
                "score": score,
                "entry": entry
            }
        )


    if not candidates:
        raise RuntimeError(
            f"No reliable YouTube result found "
            f"for {artist} - {song}"
        )


    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    # Require at least some evidence that this is correct
    if best["score"] < 10:
        raise RuntimeError(
            f"No confident YouTube match found "
            f"for {artist} - {song}"
        )

    entry = best["entry"]

    print(
        f"YouTube match score: "
        f"{best['score']}"
    )

    return {
        "title": entry["title"],
        "url": entry["webpage_url"]
    }


# ==========================================================
# RUN ALL-IN-ONE
#
# HYBRID CONFIGURATION:
#
# Demucs       -> CPU
# Spectrogram  -> CPU
# All-In-One   -> MPS
#
# ==========================================================

def run_allin1(wav_file, structure_dir):

    env = os.environ.copy()

    # Make our local ffmpeg + ffprobe available
    env["PATH"] = (
        str(Path("tools").resolve())
        + os.pathsep
        + env.get("PATH", "")
    )

    env["NATTEN_MPS"] = "0"

    print(
        "running hybrid mode: "
        "Demucs CPU + All-In-One MPS..."
    )

    subprocess.run(
        [
            "allin1",
            str(wav_file),

            "--out-dir",
            str(structure_dir),

            # Main model -> Apple GPU
            "--device",
            "mps",

            # Demucs -> CPU
            "--demucs-device",
            "cpu",

            # Spectrogram -> CPU
            "--spec-torch-device",
            "cpu",

            "--no-multiprocess"
        ],
        check=True,
        env=env
    )

    print("All-In-One hybrid SUCCESS")


# ==========================================================
# LOAD 550-SONG DATASET
# ==========================================================

df = pd.read_csv(INPUT_CSV)


# ==========================================================
# LOAD EXISTING LOG
# ==========================================================

if LOG_FILE.exists():

    log_df = pd.read_csv(LOG_FILE)

else:

    log_df = pd.DataFrame(
        columns=[
            "artist",
            "song",
            "hit",
            "youtube_title",
            "youtube_url",
            "chorus_start",
            "chorus_end",
            "chorus_file",
            "status"
        ]
    )


# ==========================================================
# FIND COMPLETED SONGS
# ==========================================================

completed = set(
    zip(
        log_df.loc[
            log_df["status"] == "success",
            "artist"
        ],
        log_df.loc[
            log_df["status"] == "success",
            "song"
        ]
    )
)


print("\n========================================")
print("DATASET PROCESSING")
print("========================================")

print("already completed:", len(completed))
print("total dataset:", len(df))


# ==========================================================
# PROCESS SONGS
# ==========================================================

for dataset_index, row in df.iterrows():

    artist = row["artist"]
    song = row["song"]
    hit = row["hit"]

    key = (artist, song)

    print("\n========================================")
    print(f"{dataset_index + 1}/{len(df)}")
    print(f"{artist} - {song}")
    print("========================================")


    # ------------------------------------------------------
    # SKIP ALREADY SUCCESSFUL SONGS
    # ------------------------------------------------------

    if key in completed:

        print(
            "already successfully processed - skipping"
        )

        continue


    base = safe_name(
        f"{artist}_{song}"
    )

    mp3_file = (
        SONGS_DIR /
        f"{base}.mp3"
    )

    wav_file = (
        SONGS_DIR /
        f"{base}.wav"
    )

    chorus_file = (
        CHORUS_DIR /
        f"{base}.wav"
    )

    structure_file = (
        STRUCTURE_DIR /
        f"{base}.json"
    )


    result_row = {

        "artist": artist,
        "song": song,
        "hit": hit,

        "youtube_title": "",
        "youtube_url": "",

        "chorus_start": "",
        "chorus_end": "",

        "chorus_file": "",

        "status": ""
    }


    try:

        # ==================================================
        # 1. SEARCH YOUTUBE
        # ==================================================

        print("searching YouTube...")

        yt_info = get_youtube_info(
            artist,
            song
        )

        print(
            "selected:",
            yt_info["title"]
        )

        result_row["youtube_title"] = (
            yt_info["title"]
        )

        result_row["youtube_url"] = (
            yt_info["url"]
        )


        # ==================================================
        # 2. DOWNLOAD MP3
        # ==================================================

        if not mp3_file.exists():

            print("downloading audio...")

            subprocess.run(
                [
                    "yt-dlp",

                    yt_info["url"],

                    "-x",

                    "--audio-format",
                    "mp3",

                    "--ffmpeg-location",
                    "./tools",

                    "-o",
                    str(mp3_file)
                ],
                check=True
            )

        else:

            print(
                "MP3 already exists - reusing it"
            )


        # ==================================================
        # 3. CONVERT MP3 -> WAV
        # ==================================================

        if not wav_file.exists():

            print("converting to wav...")

            subprocess.run(
                [
                    "./tools/ffmpeg",

                    "-y",

                    "-i",
                    str(mp3_file),

                    str(wav_file)
                ],
                check=True
            )

        else:

            print(
                "WAV already exists - reusing it"
            )


        # ==================================================
        # 4. ALL-IN-ONE
        # ==================================================

        if not structure_file.exists():

            run_allin1(
                wav_file,
                STRUCTURE_DIR
            )

        else:

            print(
                "structure JSON already exists "
                "- reusing it"
            )


        # ==================================================
        # 5. READ STRUCTURE JSON
        # ==================================================

        if not structure_file.exists():

            raise RuntimeError(
                "All-In-One output JSON not found"
            )


        with open(
            structure_file,
            "r"
        ) as f:

            structure = json.load(f)


        # ==================================================
        # 6. FIND CHORUSES
        # ==================================================

        choruses = [

            segment

            for segment
            in structure["segments"]

            if (
                segment["label"].lower()
                == "chorus"
            )
        ]


        if not choruses:

            raise RuntimeError(
                "No chorus detected"
            )


        # ==================================================
        # 7. FIRST CHORUS
        # ==================================================

        first_chorus = choruses[0]

        start = first_chorus["start"]
        end = first_chorus["end"]

        result_row["chorus_start"] = start
        result_row["chorus_end"] = end


        duration = min(
            15,
            end - start
        )


        print(
            f"first chorus: "
            f"{start:.2f}s - "
            f"{end:.2f}s"
        )


        # ==================================================
        # 8. EXTRACT CHORUS
        # ==================================================

        subprocess.run(
            [
                "./tools/ffmpeg",

                "-y",

                "-ss",
                str(start),

                "-i",
                str(wav_file),

                "-t",
                str(duration),

                str(chorus_file)
            ],
            check=True
        )


        result_row["chorus_file"] = (
            str(chorus_file)
        )

        result_row["status"] = "success"


        print("SUCCESS")


        # ==================================================
        # 9. DELETE FULL AUDIO
        # ==================================================

        if mp3_file.exists():
            mp3_file.unlink()

        if wav_file.exists():
            wav_file.unlink()


    except Exception as e:

        print(
            "FAILED:",
            e
        )

        result_row["status"] = (
            f"failed: {e}"
        )


    # ======================================================
    # UPDATE LOG IMMEDIATELY
    # ======================================================

    if len(log_df) > 0:

        log_df = log_df[
            ~(
                (
                    log_df["artist"]
                    == artist
                )
                &
                (
                    log_df["song"]
                    == song
                )
            )
        ]


    log_df = pd.concat(
        [
            log_df,
            pd.DataFrame(
                [result_row]
            )
        ],
        ignore_index=True
    )


    log_df.to_csv(
        LOG_FILE,
        index=False
    )


    # Add newly successful song to completed set
    if result_row["status"] == "success":

        completed.add(key)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n========================================")
print("PROCESSING COMPLETE")
print("========================================")


print(
    log_df["status"]
    .value_counts()
)


success_count = (
    log_df["status"]
    == "success"
).sum()


print(
    f"\nsuccessfully processed: "
    f"{success_count}/{len(df)}"
)