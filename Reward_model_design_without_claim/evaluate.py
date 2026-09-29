"""Evaluate paragraph predictions against judged exact-offset gold spans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retracted_reward.data import load_judged_data, split_by_answer


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall,
            "f1": safe_divide(2 * precision * recall, precision + recall)}


def binary_classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    positive = prf(tp, fp, fn)
    negative = prf(tn, fn, fp)
    total = tp + fp + fn + tn
    return {
        **positive,
        "true_negative": tn,
        "accuracy": safe_divide(tp + tn, total),
        "positive_class": positive,
        "negative_class": negative,
    }


def character_set(spans: list[dict]) -> set[int]:
    positions: set[int] = set()
    for span in spans:
        positions.update(range(int(span["start"]), int(span["end"])))
    return positions


def load_gold(path: str | Path, split: str, seed: int) -> dict[str, dict]:
    _claims, answers = load_judged_data(path)
    partitions = split_by_answer(answers, seed)
    selected = answers if split == "all" else partitions[split]
    return {f"{answer.paper_id}:{answer.sample_type}:{answer.answer_id}": {
        "paragraph": answer.text,
        "spans": [{"start": int(s["start"]), "end": int(s["end"]),
                   "text": str(s["text"])} for s in answer.spans]
    } for answer in selected}


def load_predictions(path: str | Path) -> dict[str, dict]:
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = root.get("predictions") if isinstance(root, dict) else None
    if not isinstance(rows, list):
        raise ValueError("prediction file must contain a predictions list")
    output = {}
    for row in rows:
        paragraph_id = str(row["paragraph_id"])
        if paragraph_id in output:
            raise ValueError(f"duplicate prediction ID: {paragraph_id}")
        spans = row.get("spans", [])
        if not isinstance(spans, list):
            raise ValueError(f"{paragraph_id}: spans must be a list")
        output[paragraph_id] = {"uses_retracted_claim": bool(
            row.get("uses_retracted_claim", spans)), "spans": spans}
    return output


def validate_spans(paragraph_id: str, paragraph: str,
                   spans: list[dict]) -> list[dict]:
    valid = []
    for span in spans:
        try:
            start, end, text = int(span["start"]), int(span["end"]), str(span["text"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{paragraph_id}: malformed prediction span") from error
        if not 0 <= start < end <= len(paragraph):
            raise ValueError(f"{paragraph_id}: span [{start}, {end}) is out of bounds")
        if paragraph[start:end] != text:
            raise ValueError(f"{paragraph_id}: span text does not match offsets")
        valid.append({"start": start, "end": end, "text": text})
    # Duplicate predictions must not inflate false-positive counts.
    return list({(s["start"], s["end"], s["text"]): s for s in valid}.values())


def evaluate(gold: dict[str, dict], predictions: dict[str, dict]) -> dict:
    missing = sorted(set(gold) - set(predictions))
    unexpected = sorted(set(predictions) - set(gold))
    binary_tp = binary_fp = binary_fn = binary_tn = 0
    char_tp = char_fp = char_fn = 0
    macro_char_precision, macro_char_recall, macro_char_f1 = [], [], []
    macro_iou, details = [], []
    for paragraph_id, target in gold.items():
        prediction = predictions.get(paragraph_id, {"uses_retracted_claim": False,
                                                     "spans": []})
        predicted_spans = validate_spans(paragraph_id, target["paragraph"],
                                         prediction["spans"])
        gold_spans = target["spans"]
        gold_positive, predicted_positive = bool(gold_spans), bool(predicted_spans)
        if gold_positive and predicted_positive: binary_tp += 1
        elif not gold_positive and predicted_positive: binary_fp += 1
        elif gold_positive and not predicted_positive: binary_fn += 1
        else: binary_tn += 1

        gold_chars, predicted_chars = character_set(gold_spans), character_set(predicted_spans)
        intersection = len(gold_chars & predicted_chars)
        union = len(gold_chars | predicted_chars)
        local_tp, local_fp, local_fn = (intersection,
            len(predicted_chars - gold_chars), len(gold_chars - predicted_chars))
        char_tp += local_tp; char_fp += local_fp; char_fn += local_fn
        local_character = prf(local_tp, local_fp, local_fn)
        # Two empty sets are a perfect local match; normally gold is positive.
        if not gold_chars and not predicted_chars:
            local_character.update(
                {"precision": 1.0, "recall": 1.0, "f1": 1.0})
        local_iou = safe_divide(intersection, union) if union else 1.0
        macro_char_precision.append(float(local_character["precision"]))
        macro_char_recall.append(float(local_character["recall"]))
        macro_char_f1.append(float(local_character["f1"]))
        macro_iou.append(local_iou)
        details.append({"paragraph_id": paragraph_id,
            "gold_positive": gold_positive, "predicted_positive": predicted_positive,
            "gold_span_count": len(gold_spans),
            "predicted_span_count": len(predicted_spans),
            "character_precision": local_character["precision"],
            "character_recall": local_character["recall"],
            "character_f1": local_character["f1"],
            "character_iou": local_iou})

    binary = binary_classification_metrics(
        binary_tp, binary_fp, binary_fn, binary_tn)
    return {"number_of_gold_examples": len(gold),
        "number_of_prediction_examples": len(predictions),
        "missing_prediction_ids": missing, "unexpected_prediction_ids": unexpected,
        "binary_detection": binary,
        "character_overlap_micro": prf(char_tp, char_fp, char_fn),
        "character_overlap_macro": {
            "mean_precision": safe_divide(
                sum(macro_char_precision), len(macro_char_precision)),
            "mean_recall": safe_divide(
                sum(macro_char_recall), len(macro_char_recall)),
            "mean_f1": safe_divide(sum(macro_char_f1), len(macro_char_f1)),
            "mean_iou": safe_divide(sum(macro_iou), len(macro_iou))},
        "per_example": details}


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="claims_and_answers_judged_large_neg.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"),
                        default="test")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = evaluate(load_gold(args.gold, args.split, args.seed),
                      load_predictions(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    summary = {"binary_detection": report["binary_detection"],
               "character_overlap_micro": report["character_overlap_micro"],
               "character_overlap_macro": report["character_overlap_macro"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
