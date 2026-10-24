from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audit = pd.read_csv("outputs/audit/image_audit.csv")
valid = audit[audit["status"] == "ok"].copy()

if valid.empty:
    raise RuntimeError("No readable images found in the audit.")

classes = sorted(valid["class"].unique())

fig, axes = plt.subplots(
    len(classes), 5,
    figsize=(15, 3.5 * len(classes)),
    squeeze=False,
)

for row_index, class_name in enumerate(classes):
    class_images = valid[valid["class"] == class_name]
    samples = class_images.sample(
        n=min(5, len(class_images)),
        random_state=42,
    )

    for ax in axes[row_index]:
        ax.axis("off")

    for column_index, (_, record) in enumerate(samples.iterrows()):
        with Image.open(DATA_DIR / record["path"]) as image:
            axes[row_index, column_index].imshow(image.convert("RGB"))

        axes[row_index, column_index].set_title(
            f"{class_name}\n"
            f"{int(record['width'])} × {int(record['height'])}"
        )

fig.suptitle("Random dataset samples by class")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sample_grid.png", dpi=180)
plt.close(fig)

valid["file_size_kib"] = valid["file_size_bytes"] / 1024

measurements = [
    ("width", "Image width", "Width (pixels)"),
    ("height", "Image height", "Height (pixels)"),
    ("file_size_kib", "File size", "File size (KiB)"),
]

for column, title, x_label in measurements:
    values_by_class = [
        valid.loc[valid["class"] == class_name, column].to_numpy()
        for class_name in classes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values_by_class,
        bins=30,
        label=classes,
        alpha=0.65,
    )

    ax.set_title(f"{title} distribution by class")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of images")
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / f"{column}_histogram.png", dpi=180)
    plt.close(fig)

print("EDA figures saved to:", OUTPUT_DIR.resolve())