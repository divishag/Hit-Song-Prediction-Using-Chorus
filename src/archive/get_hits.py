import requests
import pandas as pd

url = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/recent.json"

data = requests.get(url).json()

songs = []

for entry in data["data"][:10]:
    songs.append({
        "artist": entry["artist"],
        "song": entry["song"],
        "hit": 1
    })

df = pd.DataFrame(songs)

df.to_csv("data/hit_songs.csv", index=False)

print(df)
print("\nsaved to data/hit_songs.csv")