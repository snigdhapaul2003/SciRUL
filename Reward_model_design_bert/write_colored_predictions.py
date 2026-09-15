"""Create a colored HTML report comparing gold and predicted character spans."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from retracted_claim_pipeline import load_pairs
from write_highlighted_predictions import character_mask


CLASSES = {
    (True, True): "both",
    (True, False): "gold-only",
    (False, True): "pred-only",
}


def colored_text(paragraph: str, gold_spans: list[dict], predicted_spans: list[dict]) -> str:
    gold = character_mask(len(paragraph), gold_spans)
    predicted = character_mask(len(paragraph), predicted_spans)
    parts: list[str] = []
    index = 0
    while index < len(paragraph):
        state = (gold[index], predicted[index])
        end = index + 1
        while end < len(paragraph) and (gold[end], predicted[end]) == state:
            end += 1
        text = html.escape(paragraph[index:end])
        css_class = CLASSES.get(state)
        parts.append(f'<mark class="{css_class}">{text}</mark>' if css_class else text)
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
        default=here / "runs" / "9fold_cv" / "fold_5" / "colored_gold_vs_predicted.html",
    )
    args = parser.parse_args()

    document = json.loads(args.predictions.read_text(encoding="utf-8"))
    paragraphs = {
        item["paragraph_id"]: item["paragraph"]
        for item in document["paragraph_predictions"]
    }
    claim_texts = {item.claim_id: item.claim for item in load_pairs(args.dataset)}
    cases = []
    for number, prediction in enumerate(document["predictions"], 1):
        paragraph_id, claim_id = prediction["paragraph_id"], prediction["claim_id"]
        cases.append(f"""
        <section class="case">
          <h2>Case {number}: {html.escape(paragraph_id)} / {html.escape(claim_id.rsplit(':', 1)[-1])}</h2>
          <dl>
            <dt>Claim</dt><dd>{html.escape(claim_texts.get(claim_id, '[claim text unavailable]'))}</dd>
            <dt>Gold used</dt><dd>{str(bool(prediction.get('gold_used')))}</dd>
            <dt>Prediction</dt><dd>{html.escape(str(prediction.get('decision')))} (p_used={float(prediction.get('p_used', 0.0)):.4f})</dd>
          </dl>
          <div class="paragraph">{colored_text(paragraphs[paragraph_id], prediction.get('gold_spans', []), prediction.get('spans', []))}</div>
        </section>""")

    output = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fold 5 — Gold vs predicted spans</title>
<style>
  :root {{ color-scheme: light dark; font-family: Arial, sans-serif; }}
  body {{ max-width: 1100px; margin: 0 auto; padding: 24px; line-height: 1.55; }}
  h1 {{ margin-bottom: 8px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 28px; }}
  .legend span, mark {{ padding: 2px 5px; border-radius: 3px; color: #111; }}
  .both {{ background: #83e6a7; }}
  .gold-only {{ background: #ff8d8d; }}
  .pred-only {{ background: #ffc66d; }}
  .case {{ border-top: 1px solid #8888; padding: 22px 0; }}
  h2 {{ font-size: 1.1rem; }}
  dl {{ display: grid; grid-template-columns: 110px 1fr; gap: 4px 12px; }}
  dt {{ font-weight: bold; }} dd {{ margin: 0; }}
  .paragraph {{ white-space: pre-wrap; margin-top: 16px; font-family: Georgia, serif; }}
</style>
</head>
<body>
<h1>Fold 5: gold and predicted evidence</h1>
<div class="legend">
  <span class="both">Green: correct overlap</span>
  <span class="gold-only">Red: missed gold</span>
  <span class="pred-only">Orange: extra prediction</span>
</div>
{''.join(cases)}
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
