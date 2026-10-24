"""Build stable split manifests and TensorFlow input pipelines."""

import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from task2.common import BATCH_SIZE, IMAGE_SIZE, OUT, ROOT, SEED, write_json


def prepare_splits():
    audit_path = ROOT / "outputs/audit/image_audit.csv"
    audit = pd.read_csv(audit_path)
    valid = audit.loc[audit["status"] == "ok"].sort_values("path").copy()
    if valid["sha256"].isna().any():
        raise ValueError("Readable images must have SHA-256 hashes.")
    if valid.groupby("sha256")["class"].nunique().gt(1).any():
        raise ValueError("An identical image has conflicting class labels.")
    removed = valid.loc[valid.duplicated("sha256", keep="first")]
    valid = valid.drop_duplicates("sha256", keep="first").copy()
    valid["label"] = valid["class"].map({"Real": 0, "AI": 1})
    if valid["label"].isna().any():
        raise ValueError("Unexpected class name in audit.")
    train, held_out = train_test_split(
        valid, test_size=0.30, stratify=valid["label"], random_state=SEED,
    )
    validation, test = train_test_split(
        held_out, test_size=0.50, stratify=held_out["label"], random_state=SEED,
    )
    destination = OUT / "splits"
    destination.mkdir(parents=True, exist_ok=True)
    removed.to_csv(destination / "excluded_duplicates.csv", index=False)
    summaries = []
    split_frames = {}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame = frame.sort_values("path").reset_index(drop=True)
        frame.to_csv(destination / f"{name}.csv", index=False)
        split_frames[name] = frame
        summaries.append({"split": name, "Real": int((frame.label == 0).sum()),
                          "AI": int((frame.label == 1).sum()), "total": len(frame)})
    check_splits(split_frames)
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination / "counts.csv", index=False)
    write_json(destination / "metadata.json", {
        "seed": SEED, "fractions": [0.70, 0.15, 0.15],
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "excluded_extra_copies": len(removed), "unique_readable_images": len(valid),
        "labels": {"Real": 0, "AI": 1},
        "limitation": "Exact hashes only; near-duplicates and common sources may remain.",
    })
    print(summary.to_string(index=False))
    return split_frames


def check_splits(frames):
    seen = set()
    for name, frame in frames.items():
        hashes = set(frame["sha256"])
        if len(hashes) != len(frame) or seen.intersection(hashes):
            raise ValueError(f"Duplicate leakage in split: {name}")
        seen.update(hashes)
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"Both classes are required in {name}.")


def read_split(name):
    return pd.read_csv(OUT / "splits" / f"{name}.csv")


def decode_resize(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    # Letterboxing preserves aspect ratio. Cache uint8, then cast before augmentation.
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE, antialias=True)
    return tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8), tf.cast(label, tf.float32)


def augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant", seed=SEED + 1),
        tf.keras.layers.RandomZoom(0.10, fill_mode="constant", seed=SEED + 2),
        tf.keras.layers.RandomContrast(0.10, seed=SEED + 3),
    ], name="training_augmentation")


def make_dataset(frame, training=False, augment=True):
    paths = [str(ROOT / "dataset" / path) for path in frame["path"]]
    labels = frame["label"].to_numpy(dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(decode_resize, num_parallel_calls=2).cache()
    if training:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=2)
    if training and augment:
        transform = augmentation()
        dataset = dataset.map(
            lambda x, y: (tf.clip_by_value(transform(x, training=True), 0, 255), y),
            num_parallel_calls=1,
        )
    return dataset.prefetch(1)


def save_augmented_batch(dataset):
    images, labels = next(iter(dataset))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title("AI" if labels[i].numpy() == 1 else "Real")
        ax.axis("off")
    fig.suptitle("Task 2.1 — augmented training batch (160 × 160)")
    fig.tight_layout()
    destination = OUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "augmented_batch.png", dpi=160)
    plt.close(fig)

