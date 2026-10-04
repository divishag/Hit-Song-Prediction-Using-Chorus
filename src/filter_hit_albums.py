import pandas as pd

INPUT_FILE = "data/clean_labeled_songs.csv"
OUTPUT_FILE = "data/final_candidate_songs.csv"

df = pd.read_csv(INPUT_FILE)

# Find albums that contain at least one hit song
hit_albums = (
    df.groupby(["artist", "album"])["hit"]
      .max()
      .reset_index()
)

hit_albums = hit_albums[hit_albums["hit"] == 1][["artist", "album"]]

# Keep only songs from those albums
filtered = df.merge(
    hit_albums,
    on=["artist", "album"],
    how="inner"
)

filtered.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nclass counts:\n")
print(filtered["hit"].value_counts())

print("\ncounts by artist and class:\n")
print(
    filtered.groupby(["artist", "hit"])
            .size()
)

print("\nnumber of retained albums:")
print(
    filtered[["artist", "album"]]
    .drop_duplicates()
    .groupby("artist")
    .size()
)

print("\ntotal candidate songs:", len(filtered))

print(f"\nsaved to {OUTPUT_FILE}")