"""Print saved evidence; never train or predict from this command."""

import getpass
import json
from datetime import datetime
from task2.common import OUT, ROOT, git_commit

import pandas as pd


def main():
    print("Project: 23L-0630_A2_Ops")
    print("User:", getpass.getuser())
    print("Time:", datetime.now().astimezone().isoformat(timespec="seconds"))
    print("Directory:", ROOT)
    print("Current commit:", git_commit())
    for title, relative in [
        ("2.1 SPLITS", "splits/counts.csv"),
        ("2.5 TEST METRICS", "evaluation/metrics.csv"),
        ("2.7 FASHION-MNIST TOY", "tuning/fashion_trials.csv"),
        ("2.7 VISIONGUARD TRIALS", "tuning/vision_trials.csv"),
    ]:
        path = OUT / relative
        print(f"\n{title}")
        print(pd.read_csv(path).to_string(index=False) if path.exists() else "Not completed yet.")
    print("\n2.4 BEST EPOCHS")
    for path in sorted((OUT / "histories").glob("*.json")):
        print(path.stem, json.loads(path.read_text()))
    for relative in ["selection.json", "tuning/fashion_best.json", "tuning/vision_best.json"]:
        path = OUT / relative
        if path.exists():
            print(f"\n{relative}\n{path.read_text()}")
    summary = OUT / "summaries/baseline.txt"
    if summary.exists():
        print("\n2.2 SAVED BASELINE MODEL SUMMARY")
        print(summary.read_text())


if __name__ == "__main__":
    main()
