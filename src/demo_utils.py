import json
import os
import shutil
import subprocess
from pathlib import Path
import re

import librosa
import numpy as np
from scipy.stats import skew, kurtosis


def summarize_feature(feature_matrix):
    values = []
    for row in feature_matrix:
        values.extend([
            np.min(row),
            np.mean(row),
            np.median(row),
            np.max(row),
            np.std(row),
            skew(row),
            kurtosis(row),
        ])
    return values


def extract_features(audio_file):
    """Extract the same 518 acoustic features used by the training pipeline."""
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    features = []

    features.extend(summarize_feature(librosa.feature.chroma_stft(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.chroma_cqt(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.chroma_cens(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)))
    features.extend(summarize_feature(librosa.feature.rms(y=y)))
    features.extend(summarize_feature(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.spectral_contrast(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    harmonic = librosa.effects.harmonic(y)
    features.extend(summarize_feature(librosa.feature.tonnetz(y=harmonic, sr=sr)))
    features.extend(summarize_feature(librosa.feature.zero_crossing_rate(y)))

    features = np.asarray(features, dtype=float)

    if len(features) != 518:
        raise RuntimeError(f"Expected 518 features, got {len(features)}")
    if not np.isfinite(features).all():
        raise RuntimeError("Feature extraction produced NaN or infinite values")

    return features


def _ffmpeg_executable(project_root):
    local = Path(project_root) / "tools" / "ffmpeg"
    if local.exists():
        return str(local)
    system = shutil.which("ffmpeg")
    if system:
        return system
    raise RuntimeError("ffmpeg not found. Put it in ./tools/ffmpeg or add ffmpeg to PATH.")


def _run_allin1(wav_file, structure_dir, project_root):
    env = os.environ.copy()
    tools_dir = Path(project_root) / "tools"
    if tools_dir.exists():
        env["PATH"] = str(tools_dir.resolve()) + os.pathsep + env.get("PATH", "")
    env["NATTEN_MPS"] = "0"

    cmd = [
        "allin1",
        str(wav_file),
        "--out-dir",
        str(structure_dir),
        "--device",
        "mps",
        "--demucs-device",
        "cpu",
        "--spec-torch-device",
        "cpu",
        "--no-multiprocess",
    ]

    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"All-In-One failed. {details[-1200:]}") from exc


def extract_first_chorus(full_song_file, work_dir, project_root):
    """Detect the first chorus and save at most its first 15 seconds as WAV."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = work_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg_executable(project_root)
    wav_file = work_dir / "uploaded_song.wav"
    chorus_file = work_dir / "detected_chorus.wav"

    subprocess.run(
        [ffmpeg, "-y", "-i", str(full_song_file), str(wav_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    _run_allin1(wav_file, structure_dir, project_root)

    structure_file = structure_dir / f"{wav_file.stem}.json"
    if not structure_file.exists():
        json_files = list(structure_dir.glob("*.json"))
        if len(json_files) == 1:
            structure_file = json_files[0]
        else:
            raise RuntimeError("All-In-One finished, but its structure JSON was not found.")

    with open(structure_file, "r", encoding="utf-8") as f:
        structure = json.load(f)

    choruses = [
        segment for segment in structure.get("segments", [])
        if str(segment.get("label", "")).lower() == "chorus"
    ]
    if not choruses:
        raise RuntimeError("No chorus was detected in this song.")

    first = choruses[0]
    start = float(first["start"])
    end = float(first["end"])
    duration = min(15.0, end - start)

    if duration <= 0:
        raise RuntimeError("Detected chorus had an invalid duration.")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(start),
            "-i",
            str(wav_file),
            "-t",
            str(duration),
            str(chorus_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return chorus_file, start, start + duration

def normalize_name(text):
    text = Path(text).stem.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_cached_chorus(uploaded_filename, project_root):
    chorus_dir = Path(project_root) / "data" / "choruses"

    if not chorus_dir.exists():
        return None

    uploaded_key = normalize_name(uploaded_filename)

    for chorus_file in chorus_dir.glob("*.wav"):
        chorus_key = normalize_name(chorus_file.name)

        if uploaded_key == chorus_key:
            return chorus_file

        if uploaded_key in chorus_key or chorus_key in uploaded_key:
            return chorus_file

    return None