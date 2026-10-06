import pandas as pd
from pathlib import Path

INPUT = "data/initial_550_songs.csv"
LOG = "data/download_extract_log.csv"
OUTPUT = "data/final_548_songs.csv"

df = pd.read_csv(INPUT)
log = pd.read_csv(LOG)

# Only successful processing records
success = log[log["status"] == "success"].copy()

# Keep only songs belonging to our selected 550
final = df.merge(
    success[
        [
            "artist",
            "song",
            "youtube_title",
            "youtube_url",
            "chorus_start",
            "chorus_end",
            "chorus_file"
        ]
    ],
    on=["artist", "song"],
    how="inner"
)

# Remove accidental duplicate log entries
final = final.drop_duplicates(
    subset=["artist", "song"]
)

print("\nFinal dataset size:", len(final))

# --------------------------------------------------
# VERIFY CHORUS FILES
# --------------------------------------------------

missing_files = []

for _, row in final.iterrows():
    path = Path(row["chorus_file"])

    if not path.exists():
        missing_files.append(
            f"{row['artist']} - {row['song']}"
        )

print("Missing chorus WAV files:", len(missing_files))

for song in missing_files:
    print("MISSING:", song)


# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------

print("\nHit / non-hit:")
print(final["hit"].value_counts())

print("\nSongs per artist:")
print(final["artist"].value_counts())

print("\nArtist + class:")
print(
    final.groupby(["artist", "hit"]).size()
)


# --------------------------------------------------
# SAVE ONLY IF VALID
# --------------------------------------------------

if len(final) == 548 and len(missing_files) == 0:

    final.to_csv(
        OUTPUT,
        index=False
    )

    print(
        "\n✅ FINAL DATASET FROZEN:"
        "\ndata/final_548_songs.csv"
    )

else:

    print("\n⚠️ Dataset not frozen.")

    print(
        "Expected 548 valid songs "
        "with 548 existing chorus files."
    )