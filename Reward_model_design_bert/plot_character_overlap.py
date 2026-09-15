"""Plot character overlap, missed gold, and extra predicted evidence per case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_character_overlap import covered_characters


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "test_predictions.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "character_overlap.png",
    )
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for prediction in document["predictions"]:
        gold = covered_characters(prediction.get("gold_spans"))
        identified = covered_characters(prediction.get("spans"))
        overlap = len(gold & identified)
        missed = len(gold - identified)
        extra = len(identified - gold)
        rows.append({
            "case": f'{prediction["paragraph_id"].split(":", 1)[0]} / {prediction["claim_id"].rsplit(":", 1)[-1]}',
            "paragraph_id": prediction["paragraph_id"],
            "claim_id": prediction["claim_id"],
            "decision": prediction["decision"],
            "gold_characters": len(gold),
            "identified_characters": len(identified),
            "overlapping_characters": overlap,
            "missed_gold_characters": missed,
            "extra_predicted_characters": extra,
            "character_recall": overlap / max(1, len(gold)),
            "character_iou": overlap / max(1, len(gold | identified)),
        })

    labels = [row["case"] for row in rows]
    overlap = np.asarray([row["overlapping_characters"] for row in rows])
    missed = np.asarray([row["missed_gold_characters"] for row in rows])
    extra = np.asarray([row["extra_predicted_characters"] for row in rows])
    positions = np.arange(len(rows))

    figure, axis = plt.subplots(figsize=(13, 18), constrained_layout=True)
    axis.barh(positions, overlap, label="Correct overlap", color="#2e8b57")
    axis.barh(positions, missed, left=overlap, label="Missed gold", color="#d95f5f")
    axis.barh(
        positions, extra, left=overlap + missed,
        label="Extra prediction", color="#e6a23c",
    )
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Characters")
    axis.set_ylabel("Test case: paper / claim")
    axis.set_title("Fold 5: character-level evidence overlap by case")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
