import requests
from collections import defaultdict
from datetime import date, timedelta

START_YEAR = 2011
END_YEAR = 2025

quarter_months = [3, 6, 9, 12]


def previous_saturday(year, month):
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)

    while d.weekday() != 5:
        d -= timedelta(days=1)

    return d


artist_hits = defaultdict(set)

for year in range(START_YEAR, END_YEAR + 1):
    for month in quarter_months:

        chart_date = previous_saturday(year, month)

        url = (
            "https://raw.githubusercontent.com/"
            "mhollingshead/billboard-hot-100/main/date/"
            f"{chart_date}.json"
        )

        response = requests.get(url)

        if response.status_code != 200:
            print("missing:", chart_date)
            continue

        data = response.json()["data"]

        print("processing:", chart_date)

        for entry in data:
            artist = entry["artist"]
            song = entry["song"]

            artist_hits[artist].add(song)


ranked = sorted(
    artist_hits.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

print("\ntop artists:\n")

for artist, songs in ranked[:7]:
    print(f"{artist}: {len(songs)} hit songs")