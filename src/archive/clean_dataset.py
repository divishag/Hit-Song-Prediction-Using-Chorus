import pandas as pd
import re

INPUT_FILE = "data/labeled_songs.csv"
OUTPUT_FILE = "data/clean_labeled_songs.csv"

df = pd.read_csv(INPUT_FILE)

# Obvious alternate/non-studio track markers we don't want as separate samples.
exclude_patterns = [
    r"\blive\b",
    r"\bacoustic\b",
    r"\bremix\b",
    r"\bkaraoke\b",
    r"\bdemo\b",
    r"\bvoice memo\b",
    r"\binterlude\b",
    r"\btrack/vocal\b",
    r"\bguitar/vocal\b",
    r"\bpiano/vocal\b",
    r"\btaylor['’]s version\b",
    r"\bfrom the vault\b",
]

def should_exclude(song):
    song_lower = str(song).lower()

    for pattern in exclude_patterns:
        if re.search(pattern, song_lower):
            return True

    return False


# Remove obvious alternate-version tracks
df = df[~df["song"].apply(should_exclude)].copy()

# Remove exact duplicate rows
df = df.drop_duplicates(
    subset=["artist", "album", "song"]
)

# Save cleaned dataset
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nclass counts:\n")
print(df["hit"].value_counts())

print("\ncounts by artist and class:\n")
print(
    df.groupby(["artist", "hit"])
      .size()
)

print("\ntotal cleaned songs:", len(df))

print(f"\nsaved to {OUTPUT_FILE}")