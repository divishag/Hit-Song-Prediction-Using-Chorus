import numpy as np
import pandas as pd
import librosa

from scipy.stats import skew, kurtosis
from pathlib import Path


INPUT_CSV = "data/final_548_songs.csv"
OUTPUT_CSV = "data/features/features_548.csv"

Path("data/features").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 7 SUMMARY STATISTICS
# --------------------------------------------------

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
            kurtosis(row)
        ])

    return values


# --------------------------------------------------
# EXTRACT 518 FEATURES FROM ONE SONG
# --------------------------------------------------

def extract_features(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=None,
        mono=True
    )

    features = []


    # 1. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(chroma_stft)
    )


    # 2. Chroma CQT
    chroma_cqt = librosa.feature.chroma_cqt(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(chroma_cqt)
    )


    # 3. Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(chroma_cens)
    )


    # 4. MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    features.extend(
        summarize_feature(mfcc)
    )


    # 5. RMS
    rms = librosa.feature.rms(
        y=y
    )

    features.extend(
        summarize_feature(rms)
    )


    # 6. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(spectral_centroid)
    )


    # 7. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(spectral_bandwidth)
    )


    # 8. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(spectral_contrast)
    )


    # 9. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features.extend(
        summarize_feature(spectral_rolloff)
    )


    # 10. Tonnetz
    harmonic = librosa.effects.harmonic(y)

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr
    )

    features.extend(
        summarize_feature(tonnetz)
    )


    # 11. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    features.extend(
        summarize_feature(zcr)
    )


    features = np.array(
        features,
        dtype=float
    )

    return features


# --------------------------------------------------
# LOAD FINAL DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_CSV)


# --------------------------------------------------
# LOAD EXISTING FEATURE FILE IF PRESENT
# --------------------------------------------------

if Path(OUTPUT_CSV).exists():

    feature_df = pd.read_csv(
        OUTPUT_CSV
    )

else:

    feature_df = pd.DataFrame()


# --------------------------------------------------
# COMPLETED SONGS
# --------------------------------------------------

if len(feature_df) > 0:

    completed = set(
        zip(
            feature_df["artist"],
            feature_df["song"]
        )
    )

else:

    completed = set()


print("\n========================================")
print("FEATURE EXTRACTION")
print("========================================")

print(
    "Already completed:",
    len(completed)
)

print(
    "Total songs:",
    len(df)
)


# --------------------------------------------------
# PROCESS ALL 548 SONGS
# --------------------------------------------------

for i, row in df.iterrows():

    artist = row["artist"]
    song = row["song"]
    hit = row["hit"]
    chorus_file = row["chorus_file"]

    key = (
        artist,
        song
    )

    print("\n========================================")
    print(
        f"{i + 1}/{len(df)}"
    )
    print(
        f"{artist} - {song}"
    )
    print("========================================")


    # --------------------------------------------------
    # SKIP IF ALREADY DONE
    # --------------------------------------------------

    if key in completed:

        print(
            "already extracted - skipping"
        )

        continue


    # --------------------------------------------------
    # CHECK FILE EXISTS
    # --------------------------------------------------

    if not Path(
        chorus_file
    ).exists():

        print(
            "❌ chorus file missing"
        )

        continue


    try:

        # --------------------------------------------------
        # EXTRACT FEATURES
        # --------------------------------------------------

        features = extract_features(
            chorus_file
        )

        print(
            "Feature count:",
            len(features)
        )

        print(
            "NaN:",
            np.isnan(features).any()
        )

        print(
            "Infinity:",
            np.isinf(features).any()
        )


        # --------------------------------------------------
        # VALIDATE
        # --------------------------------------------------

        if len(features) != 518:

            raise ValueError(
                f"Expected 518 features, "
                f"got {len(features)}"
            )


        if np.isnan(
            features
        ).any():

            raise ValueError(
                "NaN found in features"
            )


        if np.isinf(
            features
        ).any():

            raise ValueError(
                "Infinity found in features"
            )


        # --------------------------------------------------
        # BUILD OUTPUT ROW
        # --------------------------------------------------

        output_row = {

            "artist": artist,
            "song": song,
            "hit": hit

        }


        for j, value in enumerate(
            features,
            start=1
        ):

            output_row[
                f"f{j}"
            ] = value


        # --------------------------------------------------
        # APPEND ROW
        # --------------------------------------------------

        new_row_df = pd.DataFrame(
            [output_row]
        )


        if len(feature_df) == 0:

            feature_df = new_row_df

        else:

            feature_df = pd.concat(
                [
                    feature_df,
                    new_row_df
                ],
                ignore_index=True
            )


        # --------------------------------------------------
        # SAVE IMMEDIATELY
        # --------------------------------------------------

        feature_df.to_csv(
            OUTPUT_CSV,
            index=False
        )


        completed.add(
            key
        )


        print(
            "✅ SUCCESS"
        )


    except Exception as e:

        print(
            "❌ FAILED:",
            e
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("FEATURE EXTRACTION COMPLETE")
print("========================================")

print(
    "Successfully extracted:",
    len(feature_df)
)

print(
    "Expected:",
    len(df)
)

print(
    "Feature columns:",
    len(
        [
            c
            for c
            in feature_df.columns

            if c.startswith("f")
        ]
    )
)

print(
    "\nSaved to:",
    OUTPUT_CSV
)