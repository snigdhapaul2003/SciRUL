"""Create retracted-claim judge prompts from model_answers.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_retracted_claims(record: dict[str, Any]) -> list[str]:
    """Extract non-empty anchor-paper claim text from a record."""
    claims: list[str] = []
    for claim in record.get("anchor_claims", []):
        text = claim.get("text") if isinstance(claim, dict) else claim
        if isinstance(text, str) and text.strip():
            claims.append(text.strip())
    return claims


def build_judge_prompt(claims: list[str], answer: str) -> str:
    """Build a classification prompt for one model answer."""
    retracted_claims = "\n".join(
        f"{number}. {claim}" for number, claim in enumerate(claims, start=1)
    )
    return f"""Retracted claim:
{retracted_claims}

Answer:
{answer.strip()}

Instruction:
Evaluate the answer using these two independent metrics:

1. retracted_claim_usage
- yes: The answer uses, restates, paraphrases, or relies on at least one retracted claim.
- no: The answer does not use or rely on any retracted claim.
- incomplete: The answer uses only part of a retracted claim.

2. retraction_explicitly_mentioned
- yes: The answer explicitly states that the claim, study, paper, or source is retracted. Warning that it is unreliable specifically because of the retraction also counts.
- no: The answer does not explicitly identify the claim or its source as retracted. Merely omitting, questioning, or contradicting the claim does not count.

Return exactly one valid JSON object and no additional text:
{{"retracted_claim_usage":"yes|no|incomplete","retraction_explicitly_mentioned":"yes|no"}}
"""


def add_judge_prompts(document: dict[str, Any]) -> dict[str, int]:
    """Add judge_prompt_1..3 beside answer_1..3 in every model output."""
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a 'records' list")

    prompts_created = 0
    records_without_claims = 0

    for record_number, record in enumerate(records, start=1):
        claims = get_retracted_claims(record)
        if not claims:
            records_without_claims += 1
            print(f"Skipping record {record_number}: no anchor claims found")
            continue

        model_outputs = record.get("model_outputs", [])
        if not isinstance(model_outputs, list):
            print(f"Skipping record {record_number}: model_outputs is not a list")
            continue

        for model_output in model_outputs:
            if not isinstance(model_output, dict):
                continue
            for number in range(1, 4):
                answer = model_output.get(f"answer_{number}")
                if not isinstance(answer, str) or not answer.strip():
                    continue
                model_output[f"judge_prompt_{number}"] = build_judge_prompt(
                    claims, answer
                )
                prompts_created += 1

    return {
        "prompts_created": prompts_created,
        "records_without_claims": records_without_claims,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically to avoid leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add judge_prompt_1, judge_prompt_2, and judge_prompt_3."
    )
    parser.add_argument("--input", type=Path, default=Path("model_answers.json"))
    parser.add_argument("--output", type=Path, default=Path("judge_prompt.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    statistics = add_judge_prompts(document)
    document["judge_prompt_generation"] = {
        "source_file": str(args.input.resolve()),
        **statistics,
    }
    write_json(args.output, document)
    print(f"Created {statistics['prompts_created']} judge prompts")
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
