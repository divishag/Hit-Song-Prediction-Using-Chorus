import pandas as pd
import requests
import re
import unicodedata

ARTISTS = [
    "Taylor Swift",
    "Morgan Wallen",
    "Drake",
    "Luke Bryan",
    "Ariana Grande",
    "Jason Aldean",
    "Ed Sheeran"
]


def normalize(text):
    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("‐", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("…", "...")
    )

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("loading discography...")

df = pd.read_csv("data/discography_tracks.csv")


print("downloading Billboard history...")

url = (
    "https://raw.githubusercontent.com/"
    "mhollingshead/billboard-hot-100/main/all.json"
)

response = requests.get(url)
response.raise_for_status()

charts = response.json()


print("building Billboard hit lookup...")

billboard_hits = {
    artist: set()
    for artist in ARTISTS
}

for chart in charts:

    for entry in chart["data"]:

        billboard_artist = entry["artist"]
        billboard_song = entry["song"]

        for artist in ARTISTS:

            if artist.lower() in billboard_artist.lower():

                billboard_hits[artist].add(
                    normalize(billboard_song)
                )


print("labeling songs...")

labels = []

for _, row in df.iterrows():

    artist = row["artist"]
    song = row["song"]

    normalized_song = normalize(song)

    if normalized_song in billboard_hits[artist]:
        labels.append(1)
    else:
        labels.append(0)


df["hit"] = labels


df.to_csv(
    "data/labeled_songs.csv",
    index=False
)


print("\nclass counts:")

print(
    df["hit"].value_counts()
)


print("\ncounts by artist:")

print(
    df.groupby(["artist", "hit"])
      .size()
)


print("\ntotal songs:", len(df))

print("\nsaved to data/labeled_songs.csv")