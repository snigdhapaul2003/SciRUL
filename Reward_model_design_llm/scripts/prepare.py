"""Create deterministic train, calibration, and test data files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retracted_claims.data import load_dataset, split_answers, to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    examples = load_dataset(args.input)
    splits = split_answers(examples, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        content = {"examples": [to_jsonable(row) for row in rows]}
        destination = output_dir / f"{split_name}.json"
        destination.write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Inference receives identifiers and paragraph text only. Claims, gold
        # spans, paper IDs, and answer-model metadata are deliberately excluded.
        inference_content = {
            "paragraphs": [
                {
                    "paragraph_id": row.paragraph_id,
                    "paragraph": row.paragraph,
                }
                for row in rows
            ]
        }
        inference_path = output_dir / f"{split_name}_inference.json"
        inference_path.write_text(
            json.dumps(inference_content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    statistics = {
        name: {
            "paragraphs": len(rows),
            "papers": len({row.paper_id for row in rows}),
        }
        for name, rows in splits.items()
    }
    statistics_path = output_dir / "split_statistics.json"
    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
