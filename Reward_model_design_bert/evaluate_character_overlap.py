"""Evaluate predicted evidence spans using gold/predicted character-set overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def covered_characters(spans: Any) -> set[int]:
    covered: set[int] = set()
    if not isinstance(spans, list):
        return covered
    for span in spans:
        if not isinstance(span, dict):
            continue
        start, end = span.get("start"), span.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
            covered.update(range(start, end))
    return covered


def safe_ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def score_example(prediction: dict[str, Any]) -> dict[str, Any]:
    gold = covered_characters(prediction.get("gold_spans"))
    predicted = covered_characters(prediction.get("spans"))
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    union = len(gold | predicted)

    # Two genuinely empty sets are a perfect localization match.
    both_empty = not gold and not predicted
    precision = safe_ratio(true_positive, len(predicted), empty_value=float(both_empty))
    recall = safe_ratio(true_positive, len(gold), empty_value=float(both_empty))
    f1 = safe_ratio(2 * precision * recall, precision + recall, empty_value=float(both_empty))
    iou = safe_ratio(true_positive, union, empty_value=float(both_empty))
    return {
        "claim_id": prediction.get("claim_id"),
        "paragraph_id": prediction.get("paragraph_id"),
        "decision": prediction.get("decision"),
        "gold_characters": len(gold),
        "predicted_characters": len(predicted),
        "overlapping_characters": true_positive,
        "false_positive_characters": false_positive,
        "false_negative_characters": false_negative,
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": f1,
        "character_iou": iou,
        "exact_character_set_match": gold == predicted,
    }


def evaluate(document: dict[str, Any]) -> dict[str, Any]:
    predictions = document.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Input must contain a non-empty 'predictions' list")
    per_example = [score_example(item) for item in predictions if isinstance(item, dict)]

    tp = sum(item["overlapping_characters"] for item in per_example)
    fp = sum(item["false_positive_characters"] for item in per_example)
    fn = sum(item["false_negative_characters"] for item in per_example)
    micro_precision = safe_ratio(tp, tp + fp)
    micro_recall = safe_ratio(tp, tp + fn)
    micro_f1 = safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "metrics": {
            "examples": len(per_example),
            "overlapping_characters": tp,
            "false_positive_characters": fp,
            "false_negative_characters": fn,
            "micro_character_precision": micro_precision,
            "micro_character_recall": micro_recall,
            "micro_character_f1": micro_f1,
            "micro_character_iou": safe_ratio(tp, tp + fp + fn),
            "macro_character_precision": mean(x["character_precision"] for x in per_example),
            "macro_character_recall": mean(x["character_recall"] for x in per_example),
            "macro_character_f1": mean(x["character_f1"] for x in per_example),
            "macro_character_iou": mean(x["character_iou"] for x in per_example),
            "exact_character_set_matches": sum(x["exact_character_set_match"] for x in per_example),
            "exact_character_set_match_rate": mean(x["exact_character_set_match"] for x in per_example),
        },
        "per_example": per_example,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=here / "runs" / "deberta" / "test_predictions.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "runs" / "deberta" / "character_overlap_metrics.json",
    )
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = evaluate(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
