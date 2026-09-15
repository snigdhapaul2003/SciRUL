"""Plot sentence-level gold/predicted evidence overlap for one CV fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from retracted_claim_pipeline import SentenceSegmenter


def touched_sentences(sentence_spans, evidence_spans):
    return {
        index
        for index, sentence in enumerate(sentence_spans)
        if any(
            int(span["start"]) < sentence.end and int(span["end"]) > sentence.start
            for span in evidence_spans
        )
    }


def build_rows(document):
    predictions = {
        (item["paragraph_id"], item["claim_id"]): item
        for item in document["predictions"]
    }
    rows = []
    segmenter = SentenceSegmenter()
    for paragraph in document["paragraph_predictions"]:
        paragraph_id = paragraph["paragraph_id"]
        sentences = segmenter(paragraph["paragraph"])
        for (candidate_paragraph, claim_id), prediction in predictions.items():
            if candidate_paragraph != paragraph_id or not prediction.get("gold_used"):
                continue
            gold = touched_sentences(sentences, prediction.get("gold_spans", []))
            identified = touched_sentences(sentences, prediction.get("spans", []))
            intersection, union = gold & identified, gold | identified
            rows.append({
                "paragraph_id": paragraph_id,
                "claim_id": claim_id,
                "claim_number": int(claim_id.rsplit("_", 1)[1]),
                "gold_sentences": len(gold),
                "identified_sentences": len(identified),
                "overlapping_sentences": len(intersection),
                "sentence_precision": len(intersection) / max(1, len(identified)),
                "sentence_recall": len(intersection) / max(1, len(gold)),
                "sentence_iou": len(intersection) / max(1, len(union)),
            })
    return rows


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "test_predictions.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "sentence_overlap.png",
    )
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_rows(document)
    paragraph_ids = sorted({row["paragraph_id"] for row in rows})
    claim_numbers = sorted({row["claim_number"] for row in rows})
    matrix = np.full((len(paragraph_ids), len(claim_numbers)), np.nan)
    labels = np.full(matrix.shape, "—", dtype=object)
    paragraph_index = {value: index for index, value in enumerate(paragraph_ids)}
    claim_index = {value: index for index, value in enumerate(claim_numbers)}
    for row in rows:
        y, x = paragraph_index[row["paragraph_id"]], claim_index[row["claim_number"]]
        matrix[y, x] = row["sentence_iou"]
        labels[y, x] = f'{row["overlapping_sentences"]}/{max(row["gold_sentences"], row["identified_sentences"])}'

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if not np.isnan(matrix[y, x]):
                axis.text(x, y, labels[y, x], ha="center", va="center", fontsize=9)
    axis.set_title("Fold 5: sentence-level evidence overlap")
    axis.set_xlabel("Gold claim")
    axis.set_ylabel("Test paragraph")
    axis.set_xticks(range(len(claim_numbers)), [f"claim_{item}" for item in claim_numbers])
    axis.set_yticks(range(len(paragraph_ids)), paragraph_ids)
    colorbar = figure.colorbar(image, ax=axis, label="Sentence IoU")
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    axis.text(
        0, -0.12,
        "Cell text = overlapping sentences / max(gold sentences, identified sentences); — = no positive gold claim",
        transform=axis.transAxes, fontsize=9,
    )
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
