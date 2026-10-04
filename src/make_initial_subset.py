import pandas as pd

INPUT = "data/final_candidate_songs.csv"
OUTPUT = "data/initial_550_songs.csv"

df = pd.read_csv(INPUT)

parts = []

# Sample proportionally within each artist AND hit/non-hit class
for (artist, hit), group in df.groupby(["artist", "hit"]):
    n = round(len(group) * 550 / len(df))
    n = min(n, len(group))

    parts.append(
        group.sample(n=n, random_state=42)
    )

subset = pd.concat(parts)

# Adjust rounding to exactly 550
if len(subset) > 550:
    subset = subset.sample(n=550, random_state=42)

elif len(subset) < 550:
    remaining = df.drop(subset.index)
    extra = remaining.sample(
        n=550 - len(subset),
        random_state=42
    )
    subset = pd.concat([subset, extra])

subset = subset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

subset.to_csv(OUTPUT, index=False)

print("total:", len(subset))
print("\nby artist:")
print(subset["artist"].value_counts())

print("\nhit/non-hit:")
print(subset["hit"].value_counts())

print("\nby artist + class:")
print(subset.groupby(["artist", "hit"]).size())