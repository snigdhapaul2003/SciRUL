"""Evaluate inference JSON against authoritative character-offset gold spans."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retracted_claims.data import load_dataset, split_answers


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(input_path: Path, prediction_path: Path, seed: int = 13) -> dict:
    prediction_root = json.loads(prediction_path.read_text(encoding="utf-8"))
    gold = {
        example.paragraph_id: example
        for example in split_answers(load_dataset(input_path), seed)["test"]
    }
    claim_catalog = {
        claim.claim_id: claim
        for example in load_dataset(input_path)
        for claim in example.claims
    }

    tp = fp = fn = tn = 0
    exact_matches = predicted_span_count = gold_span_count = 0
    intersection_chars = predicted_chars = gold_chars = 0
    abstained_claims = 0
    evaluated_paragraphs: set[str] = set()

    for prediction in prediction_root.get("predictions", []):
        paragraph_id = prediction["paragraph_id"]
        if paragraph_id not in gold:
            raise ValueError(f"Prediction {paragraph_id!r} is not in the held-out split")
        example = gold[paragraph_id]
        if prediction.get("paragraph") != example.paragraph:
            raise ValueError(f"Paragraph content mismatch for {paragraph_id}")
        evaluated_paragraphs.add(paragraph_id)

        predicted_by_claim: dict[str, set[tuple[int, int]]] = defaultdict(set)
        verdict_by_claim: dict[str, str] = {}
        # New inference writes paragraph-level aggregation. The sentence-level
        # fallback keeps old prediction files evaluable.
        claim_predictions = prediction.get("claims")
        if claim_predictions is None:
            claim_predictions = [
                claim
                for sentence in prediction.get("sentences", [])
                for claim in sentence.get("claims", [])
            ]
        for claim in claim_predictions:
                claim_id = str(claim["claim_id"])
                if claim_id not in claim_catalog:
                    raise ValueError(f"Unknown retrieved claim ID: {claim_id}")
                for step in claim.get("steps", []):
                    if step.get("status") != "verified_positive" or step.get("start") is None:
                        continue
                    start, end = int(step["start"]), int(step["end"])
                    span = str(step["span"])
                    assert example.paragraph[start:end] == span
                    predicted_by_claim[claim_id].add((start, end))
                if claim.get("verdict") in {"used", "not_used", "abstain"}:
                    verdict_by_claim[claim_id] = claim["verdict"]

        # Evaluate against the full frozen catalogue. A claim not retrieved is an
        # end-to-end negative prediction; a cross-paper attribution is an FP.
        for claim_id, claim in claim_catalog.items():
            gold_spans = set()
            if claim.paper_id == example.paper_id:
                gold_spans = {
                    (span.start, span.end)
                    for span in example.spans
                    if claim.claim_number in span.claim_numbers
                }
            predicted_spans = predicted_by_claim[claim_id]

            gold_used, predicted_used = bool(gold_spans), bool(predicted_spans)
            if verdict_by_claim.get(claim_id) == "abstain":
                abstained_claims += 1
            tp += int(gold_used and predicted_used)
            fp += int(not gold_used and predicted_used)
            fn += int(gold_used and not predicted_used)
            tn += int(not gold_used and not predicted_used)

            exact_matches += len(gold_spans & predicted_spans)
            predicted_span_count += len(predicted_spans)
            gold_span_count += len(gold_spans)

            gold_character_set = set().union(
                *(set(range(start, end)) for start, end in gold_spans)
            ) if gold_spans else set()
            predicted_character_set = set().union(
                *(set(range(start, end)) for start, end in predicted_spans)
            ) if predicted_spans else set()
            intersection_chars += len(gold_character_set & predicted_character_set)
            gold_chars += len(gold_character_set)
            predicted_chars += len(predicted_character_set)

    missing = set(gold) - evaluated_paragraphs
    if missing:
        raise ValueError(f"Missing {len(missing)} test paragraphs: {sorted(missing)}")

    usage_precision = divide(tp, tp + fp)
    usage_recall = divide(tp, tp + fn)
    exact_precision = divide(exact_matches, predicted_span_count)
    exact_recall = divide(exact_matches, gold_span_count)
    character_precision = divide(intersection_chars, predicted_chars)
    character_recall = divide(intersection_chars, gold_chars)

    return {
        "claim_paragraph_usage": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": divide(tp + tn, tp + fp + fn + tn),
            "precision": usage_precision,
            "recall": usage_recall,
            "f1": divide(2 * usage_precision * usage_recall, usage_precision + usage_recall),
        },
        "span_exact": {
            "matching": exact_matches,
            "predicted": predicted_span_count,
            "gold": gold_span_count,
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": divide(2 * exact_precision * exact_recall, exact_precision + exact_recall),
        },
        "character_overlap_micro": {
            "intersection_chars": intersection_chars,
            "predicted_chars": predicted_chars,
            "gold_chars": gold_chars,
            "precision": character_precision,
            "recall": character_recall,
            "f1": divide(2 * character_precision * character_recall,
                         character_precision + character_recall),
            "iou": divide(intersection_chars,
                          predicted_chars + gold_chars - intersection_chars),
        },
        "parse_failures": prediction_root.get("parse_failures", 0),
        "evaluated_paragraphs": len(evaluated_paragraphs),
        "catalog_claims": len(claim_catalog),
        "abstention_rate": divide(
            abstained_claims,
            len(evaluated_paragraphs) * len(claim_catalog),
        ),
        "evaluation_scope": "end-to-end paragraph x full frozen claim catalogue",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Original annotated input JSON")
    parser.add_argument("predictions", type=Path, help="Inference output JSON")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    metrics = evaluate(args.input, args.predictions, args.seed)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
