import musicbrainzngs
import pandas as pd
import time

musicbrainzngs.set_useragent(
    "hit-song-chorus-ml",
    "1.0",
    "student-project"
)

ARTISTS = [
    "Taylor Swift",
    "Morgan Wallen",
    "Drake",
    "Luke Bryan",
    "Ariana Grande",
    "Jason Aldean",
    "Ed Sheeran"
]

exclude_words = [
    "taylor's version",
    "taylor’s version",
    "deluxe",
    "live",
    "karaoke",
    "demo",
    "remix",
    "secret studio sessions"
]


def get_artist_id(artist_name):
    result = musicbrainzngs.search_artists(
        artist=artist_name,
        limit=5
    )

    for artist in result["artist-list"]:
        if artist["name"].lower() == artist_name.lower():
            return artist["id"]

    return result["artist-list"][0]["id"]


def get_studio_albums(artist_id):
    result = musicbrainzngs.browse_release_groups(
        artist=artist_id,
        release_type=["album"],
        limit=100
    )

    albums = []

    for album in result["release-group-list"]:

        title = album["title"]
        primary = album.get("primary-type")
        secondary = album.get("secondary-type-list", [])

        if primary != "Album":
            continue

        if secondary:
            continue

        if any(word in title.lower() for word in exclude_words):
            continue

        albums.append(album)

    return albums


def get_album_tracks(album):
    album_id = album["id"]

    releases = musicbrainzngs.browse_releases(
        release_group=album_id,
        limit=100
    )

    candidates = []

    for release_info in releases["release-list"]:

        if release_info.get("status") != "Official":
            continue

        release_id = release_info["id"]
        release_date = release_info.get("date", "9999-99-99")

        try:
            full_release = musicbrainzngs.get_release_by_id(
                release_id,
                includes=["recordings"]
            )["release"]

        except Exception:
            continue

        tracks = []

        for medium in full_release.get("medium-list", []):
            for track in medium.get("track-list", []):

                title = track["recording"].get("title")

                if title:
                    tracks.append(title)

        if not tracks:
            continue

        candidates.append({
            "date": release_date,
            "track_count": len(tracks),
            "tracks": tracks
        })

    if not candidates:
        return []

    candidates.sort(
        key=lambda x: (
            x["date"],
            x["track_count"]
        )
    )

    earliest_date = candidates[0]["date"]

    earliest_candidates = [
        candidate
        for candidate in candidates
        if candidate["date"] == earliest_date
    ]

    chosen = min(
        earliest_candidates,
        key=lambda x: x["track_count"]
    )

    return chosen["tracks"]


rows = []

for artist_name in ARTISTS:

    print(f"\n===== {artist_name} =====")

    artist_id = get_artist_id(artist_name)

    print("MusicBrainz ID:", artist_id)

    albums = get_studio_albums(artist_id)

    for album in albums:

        album_title = album["title"]

        tracks = get_album_tracks(album)

        if not tracks:
            print("skipping:", album_title)
            continue

        print(f"{album_title}: {len(tracks)} tracks")

        for track in tracks:
            rows.append({
                "artist": artist_name,
                "album": album_title,
                "song": track
            })

        time.sleep(0.2)


df = pd.DataFrame(rows)

df = df.drop_duplicates(
    subset=["artist", "album", "song"]
)

df.to_csv(
    "data/discography_tracks.csv",
    index=False
)

print("\n============================")
print("tracks per artist:\n")

print(
    df.groupby("artist")
      .size()
      .sort_values(ascending=False)
)

print("\ntotal album tracks:", len(df))

print("\nsaved to data/discography_tracks.csv")