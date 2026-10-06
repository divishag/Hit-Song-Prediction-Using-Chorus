import requests
import pandas as pd

START_YEAR = 2011
END_YEAR = 2025

SELECTED_ARTISTS = [
    "Taylor Swift",
    "Morgan Wallen",
    "Drake",
    "Luke Bryan",
    "Ariana Grande",
    "Jason Aldean",
    "Ed Sheeran"
]

url = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"

print("downloading Billboard Hot 100 history...")

response = requests.get(url)
response.raise_for_status()

charts = response.json()

hits = set()

for chart in charts:
    year = int(chart["date"][:4])

    if year < START_YEAR or year > END_YEAR:
        continue

    for entry in chart["data"]:
        artist = entry["artist"]
        song = entry["song"]

        for selected_artist in SELECTED_ARTISTS:
            if selected_artist.lower() in artist.lower():
                hits.add((selected_artist, song))

rows = []

for artist, song in sorted(hits):
    rows.append({
        "artist": artist,
        "song": song,
        "hit": 1
    })

df = pd.DataFrame(rows)

df.to_csv("data/artist_hits.csv", index=False)

print("\nhit-song counts:\n")
print(df.groupby("artist").size().sort_values(ascending=False))

print("\ntotal hit songs:", len(df))
print("\nsaved to data/artist_hits.csv")