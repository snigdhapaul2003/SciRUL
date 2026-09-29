"""Force all negative-answer judgments to the hard-negative label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from judge_retracted_claim_usage import update_statistics


def force_negative_no_span(document: dict[str, Any]) -> tuple[int, int]:
    total = 0
    changed = 0
    for sample in document.get("samples", []):
        negative = sample.get("negative", {})
        if not isinstance(negative, dict):
            continue
        judgments = negative.setdefault("judgments", {})
        for answer_key in negative.get("answers", {}):
            total += 1
            forced_judgment = {"uses_retracted_claim": "no", "spans": []}
            if judgments.get(answer_key) != forced_judgment:
                changed += 1
            judgments[answer_key] = forced_judgment
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    document = json.loads(args.path.read_text(encoding="utf-8-sig"))
    total, changed = force_negative_no_span(document)
    model = document.get("judge_statistics", {}).get("judge_model", "unknown")
    update_statistics(document, model)

    temporary = args.path.with_suffix(args.path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.path)
    print(f"Negative answers: {total}")
    print(f"Judgments changed: {changed}")


if __name__ == "__main__":
    main()
