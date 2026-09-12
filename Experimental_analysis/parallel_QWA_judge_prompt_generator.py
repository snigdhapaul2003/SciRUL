"""Create claim-usage judge prompts for every model answer in parallel QA output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("parallel_QA_model_answers.json")
DEFAULT_OUTPUT = Path(__file__).with_name("parallel_QWA_judge_prompts.json")
EVIDENCE_MARKER = "EVIDENCE FOR QUESTION DESIGN:"


def extract_claims(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract retracted and valid claim text from a result's design prompt."""
    prompt = result.get("question_generation_prompt")
    if not isinstance(prompt, str) or EVIDENCE_MARKER not in prompt:
        raise ValueError("question_generation_prompt has no evidence section")

    evidence_text = prompt.split(EVIDENCE_MARKER, 1)[1].lstrip()
    try:
        evidence, _ = json.JSONDecoder().raw_decode(evidence_text)
    except json.JSONDecodeError as error:
        raise ValueError("evidence section is not valid JSON") from error

    retracted_claims = [
        claim.strip()
        for claim in evidence.get("retracted_claims", [])
        if isinstance(claim, str) and claim.strip()
    ]
    valid_claims = []
    for claim in evidence.get("candidate_claims", []):
        text = claim.get("text") if isinstance(claim, dict) else claim
        if isinstance(text, str) and text.strip():
            valid_claims.append(text.strip())

    if not retracted_claims:
        raise ValueError("no retracted claims found")
    if not valid_claims:
        raise ValueError("no valid claims found")
    return retracted_claims, valid_claims


def numbered(claims: list[str]) -> str:
    return "\n".join(f"{index}. {claim}" for index, claim in enumerate(claims, 1))


def build_judge_prompt(
    retracted_claims: list[str], valid_claims: list[str], answer: str
) -> str:
    """Build a five-metric, yes/no classification prompt for one answer."""
    return f"""You are judging which supplied evidence an answer uses.

RETRACTED PAPER CLAIMS:
{numbered(retracted_claims)}

VALID PAPER CLAIMS:
{numbered(valid_claims)}

MODEL ANSWER:
{answer.strip()}

INSTRUCTIONS:
Judge semantic use, including direct statements, close paraphrases, and reasoning that relies
on a claim. Do not count a coincidental shared word or broad topic as claim use. Evaluate only
against the claims supplied above; do not add outside knowledge.

Return five yes/no metrics:
1. uses_retracted_claim: yes if the answer uses at least one retracted paper claim; otherwise no.
2. uses_valid_claim: yes if the answer uses at least one valid paper claim; otherwise no.
3. uses_both: yes only if both uses_retracted_claim and uses_valid_claim are yes; otherwise no.
4. uses_none: yes only if both uses_retracted_claim and uses_valid_claim are no; otherwise no.
5. mentions_retracted_paper_as_retracted: yes only if the answer explicitly says that the
   retracted paper, its source, or its claim is retracted; otherwise no. Merely doubting,
   rejecting, or omitting a claim does not count.

The fields must be logically consistent: uses_both and uses_none cannot both be yes.
Return exactly one valid JSON object and no markdown or additional text:
{{"uses_retracted_claim":"yes|no","uses_valid_claim":"yes|no","uses_both":"yes|no","uses_none":"yes|no","mentions_retracted_paper_as_retracted":"yes|no"}}
"""


def add_judge_prompts(document: dict[str, Any]) -> dict[str, int]:
    """Add judge_prompt_1..3 beside all available model answers."""
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError("Input JSON must contain a 'results' list")

    prompts_created = 0
    results_skipped = 0
    answers_skipped = 0

    for result_number, result in enumerate(results, 1):
        if not isinstance(result, dict):
            results_skipped += 1
            print(f"Skipping result {result_number}: result is not an object")
            continue
        try:
            retracted_claims, valid_claims = extract_claims(result)
        except ValueError as error:
            results_skipped += 1
            print(f"Skipping result {result_number}: {error}")
            continue

        model_outputs = result.get("model_outputs")
        if not isinstance(model_outputs, list):
            results_skipped += 1
            print(f"Skipping result {result_number}: model_outputs is not a list")
            continue

        for model_output in model_outputs:
            if not isinstance(model_output, dict):
                continue
            for answer_number in range(1, 4):
                answer = model_output.get(f"answer_{answer_number}")
                if not isinstance(answer, str) or not answer.strip():
                    answers_skipped += 1
                    continue
                model_output[f"judge_prompt_{answer_number}"] = build_judge_prompt(
                    retracted_claims, valid_claims, answer
                )
                prompts_created += 1

    return {
        "prompts_created": prompts_created,
        "results_skipped": results_skipped,
        "answers_skipped": answers_skipped,
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create claim-usage judge prompts for parallel QA model answers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    print(f"Skipped {statistics['results_skipped']} results")
    print(f"Skipped {statistics['answers_skipped']} missing answers")
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
