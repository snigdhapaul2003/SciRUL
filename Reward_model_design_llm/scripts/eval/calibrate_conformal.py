"""Fit a finite-sample step-confidence threshold on calibration predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path, help="Prepared calibration.json")
    parser.add_argument("predictions", type=Path, help="Calibration inference output")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between zero and one")

    gold_root = json.loads(args.gold.read_text(encoding="utf-8"))
    prediction_root = json.loads(args.predictions.read_text(encoding="utf-8"))
    gold_by_paragraph = {
        example["paragraph_id"]: example
        for example in gold_root["examples"]
    }

    correct_step_confidences = []
    emitted_steps = 0
    correct_steps = 0
    for prediction in prediction_root["predictions"]:
        example = gold_by_paragraph[prediction["paragraph_id"]]
        gold_by_claim = {}
        for claim in example["claims"]:
            claim_number = int(claim["claim_number"])
            claim_id = f"{example['paper_id']}:claim_{claim_number}"
            gold_by_claim[claim_id] = {
                (int(span["start"]), int(span["end"]))
                for span in example["spans"]
                if claim_number in span["claim_numbers"]
            }

        seen = set()
        for claim_prediction in prediction["claims"]:
            claim_id = claim_prediction["claim_id"]
            for step in claim_prediction["steps"]:
                if step["status"] != "verified_positive" or step["start"] is None:
                    continue
                key = (claim_id, int(step["start"]), int(step["end"]), step["score"])
                if key in seen:
                    continue
                seen.add(key)
                emitted_steps += 1
                span_offsets = (int(step["start"]), int(step["end"]))
                if span_offsets in gold_by_claim.get(claim_id, set()):
                    correct_steps += 1
                    correct_step_confidences.append(float(step["score"]))

    if not correct_step_confidences:
        raise ValueError("no exactly correct verified steps available for calibration")

    nonconformity = sorted(1.0 - score for score in correct_step_confidences)
    sample_count = len(nonconformity)
    rank = min(sample_count, math.ceil((sample_count + 1) * (1 - args.alpha)))
    quantile = nonconformity[rank - 1]
    confidence_threshold = 1.0 - quantile
    empirical_coverage = sum(
        score >= confidence_threshold for score in correct_step_confidences
    ) / sample_count

    report = {
        "alpha": args.alpha,
        "calibration_correct_steps": sample_count,
        "all_emitted_verified_steps": emitted_steps,
        "exactly_correct_emitted_steps": correct_steps,
        "raw_step_precision": correct_steps / emitted_steps if emitted_steps else 0.0,
        "finite_sample_rank": rank,
        "nonconformity_quantile": quantile,
        "step_confidence_threshold": confidence_threshold,
        "empirical_coverage": empirical_coverage,
        "target_coverage": 1 - args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
