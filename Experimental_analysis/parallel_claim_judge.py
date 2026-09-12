import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from litellm import completion


DEFAULT_INPUT_FILE = Path(__file__).with_name("parallel_claim_prompts.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("parallel_claim_scores.json")
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "azure/gpt-5.5")
DEFAULT_MAX_RETRIES = 3

AGGREGATE_SYSTEM_PROMPT = """You evaluate whether claims from other papers are parallel to a retracted paper's claims.

Aggregate all valid claims belonging to the same source paper into ONE paper-level judgment.
Do not return claim-level scores. A source paper may contain several claims; judge their
combined evidence as a single body of work against all retracted claims.

For every source paper, provide three scores from 0 to 5:

1. same_problem_same_solution: the source paper addresses the same scientific problem,
     task, object, population, setting, or outcome and uses the same or closely related
     method, algorithm, model, or solution.

2. same_problem_different_solution: the source paper addresses the same scientific
     problem, task, object, population, setting, or outcome but uses a meaningfully different
     method, algorithm, model, or solution.

3. same_solution_different_problem: the source paper uses the same or closely related
     method, algorithm, model, or solution but addresses a meaningfully different problem,
     task, object, population, setting, or outcome.

Use these weights for the paper-level overall score:
- same_problem_same_solution: 0.50
- same_problem_different_solution: 0.30
- same_solution_different_problem: 0.20

Calculate overall_weighted_score as:
(0.50 * same_problem_same_solution_score) +
(0.30 * same_problem_different_solution_score) +
(0.20 * same_solution_different_problem_score)

Do not treat shared words, broad subject areas, or a shared algorithm alone as strong
parallelism. Consider the combined claims, scientific problem, setting, outcomes, and
methods. A paper-level score must reflect the strongest defensible relationship across its
claims without allowing one isolated claim to dominate unrelated claims.

Return only this JSON object:
{
    "paper_scores": [
        {
            "paper_rank": "1",
            "title": "source paper title",
            "claims_considered": 0,
            "same_problem_same_solution_score": 0,
            "same_problem_different_solution_score": 0,
            "same_solution_different_problem_score": 0,
            "overall_weighted_score": 0,
            "rationale": "short evidence-based explanation"
        }
    ]
}

Include exactly one result for every source paper represented in valid_claims. Use integer
component scores from 0 to 5, preserve paper_rank and title, and do not include individual
claim scores or claim-level rankings.
"""


def load_env_values(env_file: Path) -> dict[str, str]:
    values = {
        key: value
        for key, value in os.environ.items()
        if key in {"AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"}
        and value
    }
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() not in values and value.strip():
                values[key.strip()] = value.strip().strip('"')

    api_key = values.get("AZURE_API_KEY") or values.get("AZURE_OPENAI_API_KEY")
    missing = [
        name
        for name, value in (
            ("AZURE_API_KEY or AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_API_BASE", values.get("AZURE_API_BASE")),
            ("AZURE_API_VERSION", values.get("AZURE_API_VERSION")),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required Azure settings: {', '.join(missing)}")

    return {
        "api_key": api_key,
        "api_base": values["AZURE_API_BASE"],
        "api_version": values["AZURE_API_VERSION"],
    }


def parse_json_response(response: Any) -> dict[str, Any]:
    content = response.choices[0].message.content
    if not content:
        raise ValueError("the model returned an empty response")
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get("paper_scores"), list):
        raise ValueError("the model response must contain a paper_scores list")
    return payload


def score_record(
    record: dict[str, Any],
    model: str,
    azure_config: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    prompt = record.get("parallel_claim_prompt") or {}
    user_prompt = prompt.get("user_prompt")
    if not isinstance(user_prompt, str):
        raise ValueError("record has no generated parallel-claim prompt")

    completion_options: dict[str, Any] = {
        "response_format": {"type": "json_object"},
    }
    if not model.lower().split("/")[-1].startswith("gpt-5"):
        completion_options["temperature"] = 0

    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **completion_options,
                **azure_config,
            )
            return parse_json_response(response)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))

    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask GPT-5.5 to score generated parallel-claim prompts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        source = json.load(input_file)
    records = source.get("records", []) if isinstance(source, dict) else source
    selected_records = records[args.start :]
    if args.limit is not None:
        selected_records = selected_records[: args.limit]
    azure_config = load_env_values(args.input.with_name(".env"))

    output_records = []
    succeeded = 0
    failed = 0
    for index, record in enumerate(selected_records, args.start):
        output_record = {
            "record_index": index,
            "record_id": (record.get("anchor_csv_data") or {}).get("Record ID"),
            "title": (record.get("anchor_csv_data") or {}).get("Title"),
        }
        try:
            output_record["parallel_paper_scores"] = score_record(
                record,
                args.model,
                azure_config,
                args.max_retries,
            )
            output_record["status"] = "success"
            succeeded += 1
        except Exception as error:
            output_record["status"] = "error"
            output_record["error"] = f"{type(error).__name__}: {error}"
            failed += 1
        output_records.append(output_record)
        print(f"[{index + 1}/{len(records)}] {output_record['status']}: {output_record['title']}")

    output = {
        "model": args.model,
        "input_file": str(args.input),
        "results": output_records,
        "statistics": {
            "records_requested": len(selected_records),
            "records_succeeded": succeeded,
            "records_failed": failed,
            "start": args.start,
            "limit": args.limit,
        },
    }
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")
    print(f"Saved {succeeded} scores and {failed} errors to {args.output}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
