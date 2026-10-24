from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    "AiArtData": "AI",
    "RealArt": "Real",
}

rows = []

for path in sorted(DATA_DIR.rglob("*")):
    if not path.is_file():
        continue

    relative_path = path.relative_to(DATA_DIR)
    folder = relative_path.parts[0]

    if folder not in CLASS_NAMES:
        raise ValueError(f"Unexpected dataset folder: {folder}")

    row = {
        "path": str(relative_path),
        "class": CLASS_NAMES[folder],
        "file_size_bytes": None,
        "width": None,
        "height": None,
        "colour_mode": None,
        "format": None,
        "sha256": None,
        "status": "ok",
        "error": "",
    }

    try:
        row["file_size_bytes"] = path.stat().st_size

        with path.open("rb") as file:
            row["sha256"] = hashlib.file_digest(file, "sha256").hexdigest()

        with Image.open(path) as image:
            image.load()

            row["width"], row["height"] = image.size
            row["colour_mode"] = image.mode
            row["format"] = image.format

    except Exception as error:
        row["status"] = "unreadable"
        row["error"] = f"{type(error).__name__}: {error}"

    rows.append(row)

if not rows:
    raise RuntimeError("No files found. Check DATA_DIR and your working directory.")

audit = pd.DataFrame(rows)
audit.to_csv(OUTPUT_DIR / "image_audit.csv", index=False)


valid = audit[audit["status"] == "ok"]

counts = (
    audit.groupby(["class", "status"])
    .size()
    .unstack(fill_value=0)
)

statistics = valid.groupby("class")[
    ["width", "height", "file_size_bytes"]
].agg(["min", "median", "max"])

hashed = audit.dropna(subset=["sha256"])
duplicates = hashed[
    hashed.duplicated(subset="sha256", keep=False)
].sort_values(["sha256", "path"])

duplicate_groups = duplicates["sha256"].nunique()
extra_copies = len(duplicates) - duplicate_groups

cross_class_groups = (
    duplicates.groupby("sha256")["class"].nunique().gt(1).sum()
)

counts.to_csv(OUTPUT_DIR / "class_counts.csv")
statistics.to_csv(OUTPUT_DIR / "statistics.csv")
duplicates.to_csv(OUTPUT_DIR / "duplicates.csv", index=False)

audit[audit["status"] != "ok"].to_csv(
    OUTPUT_DIR / "unreadable_files.csv", index=False
)

print("\nFILE COUNTS")
print(counts.to_string())

print("\nSTATISTICS FOR READABLE IMAGES")
print(statistics.to_string())

print("\nDUPLICATE SUMMARY")
print("Groups of identical files:", duplicate_groups)
print("Files belonging to those groups:", len(duplicates))
print("Extra copies beyond one per group:", extra_copies)
print("Duplicate groups appearing in both classes:", cross_class_groups)

print("\nUnreadable files:", (audit["status"] != "ok").sum())
print("Tables saved to:", OUTPUT_DIR.resolve())