"""Write gold and predicted evidence highlights directly into target paragraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retracted_claim_pipeline import load_pairs


MARKERS = {
    (True, True): ("<<<BOTH>>>", "<<<END_BOTH>>>"),
    (True, False): ("<<<GOLD_ONLY>>>", "<<<END_GOLD_ONLY>>>"),
    (False, True): ("<<<PRED_ONLY>>>", "<<<END_PRED_ONLY>>>"),
}


def character_mask(length: int, spans: list[dict]) -> list[bool]:
    mask = [False] * length
    for span in spans:
        start = max(0, int(span["start"]))
        end = min(length, int(span["end"]))
        for index in range(start, end):
            mask[index] = True
    return mask


def highlighted(paragraph: str, gold_spans: list[dict], predicted_spans: list[dict]) -> str:
    gold = character_mask(len(paragraph), gold_spans)
    predicted = character_mask(len(paragraph), predicted_spans)
    parts: list[str] = []
    index = 0
    while index < len(paragraph):
        state = (gold[index], predicted[index])
        end = index + 1
        while end < len(paragraph) and (gold[end], predicted[end]) == state:
            end += 1
        text = paragraph[index:end]
        if state in MARKERS:
            opening, closing = MARKERS[state]
            parts.extend((opening, text, closing))
        else:
            parts.append(text)
        index = end
    return "".join(parts)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "test_predictions.json",
    )
    parser.add_argument(
        "--dataset", type=Path, default=here / "claims_and_answers_judged.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "runs" / "9fold_cv" / "fold_5" / "highlighted_gold_vs_predicted.txt",
    )
    args = parser.parse_args()

    document = json.loads(args.predictions.read_text(encoding="utf-8"))
    paragraphs = {
        item["paragraph_id"]: item["paragraph"]
        for item in document["paragraph_predictions"]
    }
    claim_texts = {item.claim_id: item.claim for item in load_pairs(args.dataset)}
    lines = [
        "FOLD 5 — GOLD VS PREDICTED CHARACTER SPANS",
        "",
        "Legend:",
        "  <<<BOTH>>> text <<<END_BOTH>>>                 gold and predicted overlap",
        "  <<<GOLD_ONLY>>> text <<<END_GOLD_ONLY>>>       missed gold text",
        "  <<<PRED_ONLY>>> text <<<END_PRED_ONLY>>>       extra predicted text",
        "",
        "=" * 100,
    ]
    for number, prediction in enumerate(document["predictions"], 1):
        paragraph_id = prediction["paragraph_id"]
        claim_id = prediction["claim_id"]
        lines.extend([
            "",
            f"CASE {number}",
            f"Paragraph: {paragraph_id}",
            f"Claim: {claim_id}",
            f"Claim text: {claim_texts.get(claim_id, '[claim text unavailable]')}",
            f"Gold used: {bool(prediction.get('gold_used'))}",
            f"Predicted decision: {prediction.get('decision')}",
            f"Predicted p_used: {float(prediction.get('p_used', 0.0)):.6f}",
            "",
            highlighted(
                paragraphs[paragraph_id],
                prediction.get("gold_spans", []),
                prediction.get("spans", []),
            ),
            "",
            "=" * 100,
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
