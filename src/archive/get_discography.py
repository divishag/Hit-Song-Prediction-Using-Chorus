import musicbrainzngs
from datetime import datetime

musicbrainzngs.set_useragent(
    "hit-song-chorus-ml",
    "1.0",
    "student-project"
)

artist_id = "20244d07-534f-4eff-b4d4-930878889970"

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

result = musicbrainzngs.browse_release_groups(
    artist=artist_id,
    release_type=["album"],
    limit=100
)

studio_albums = []

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

    studio_albums.append(album)

print("album tracks:\n")

for album in studio_albums:
    album_title = album["title"]
    album_id = album["id"]

    releases = musicbrainzngs.browse_releases(
        release_group=album_id,
        limit=100
    )

    release_list = releases["release-list"]

    candidates = []

    for release_info in release_list:

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
        print(f"\nskipping {album_title}: no suitable release found")
        continue

    candidates.sort(
        key=lambda x: (
            x["date"],
            x["track_count"]
        )
    )

    earliest_date = candidates[0]["date"]

    earliest_candidates = [
        c for c in candidates
        if c["date"] == earliest_date
    ]

    chosen = min(
        earliest_candidates,
        key=lambda x: x["track_count"]
    )

    print(f"\n--- {album_title} ---")

    for track in chosen["tracks"]:
        print(track)