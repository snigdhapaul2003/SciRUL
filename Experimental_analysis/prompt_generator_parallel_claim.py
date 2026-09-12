import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path(__file__).with_name("similar_papers_claims.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("parallel_claim_prompts.json")

SCORING_SYSTEM_PROMPT = """You evaluate whether claims from other papers are parallel to a retracted claim.

For every VALID CLAIM, provide three separate parallelism scores from 0 to 5:

1. same_problem_same_solution: the valid claim targets the same scientific problem, task,
   object, population, setting, or outcome and uses the same or closely related method,
   algorithm, model, or solution. This is the most important category.

2. same_problem_different_solution: the valid claim targets the same scientific problem,
   task, object, population, setting, or outcome but uses a meaningfully different method,
   algorithm, model, or solution.

3. same_solution_different_problem: the valid claim uses the same or closely related method,
   algorithm, model, or solution but targets a meaningfully different scientific problem,
   task, object, population, setting, or outcome.

Use these weights for the overall score:
- same_problem_same_solution: 0.50
- same_problem_different_solution: 0.30
- same_solution_different_problem: 0.20

Calculate overall_weighted_score as:
(0.50 * same_problem_same_solution_score) +
(0.30 * same_problem_different_solution_score) +
(0.20 * same_solution_different_problem_score)

Use this scale for every component score:
0 = no meaningful parallelism
1 = very weak or only topical overlap
2 = partial overlap, but important elements differ
3 = moderate parallelism
4 = strong parallelism with one notable difference
5 = very strong parallelism; the specified relationship is clear and central

Do not treat shared words alone as evidence. Judge claim content, not titles alone. The
three scores are independent. Rank all valid claims separately for each component score and
also by overall_weighted_score. Use competition ranking for ties: equal scores receive the
same rank, and the next rank is skipped, such as scores 5, 5, 3 receiving ranks 1, 1, 3.

Return only this JSON object:
{
  "claim_scores": [
    {
      "claim_id": "paper_1_claim_1",
      "same_problem_same_solution_score": 0,
      "same_problem_same_solution_rank": 1,
      "same_problem_different_solution_score": 0,
      "same_problem_different_solution_rank": 1,
      "same_solution_different_problem_score": 0,
      "same_solution_different_problem_rank": 1,
      "overall_weighted_score": 0,
      "overall_rank": 1,
      "rationale": "short evidence-based explanation"
    }
  ]
}

Include exactly one result for every VALID CLAIM, preserve each claim_id exactly, use integer
component scores and ranks, and do not include retracted claims in claim_scores.
"""


def claim_texts(claims: list[Any]) -> list[str]:
    return [
        " ".join(claim["text"].split())
        for claim in claims
        if isinstance(claim, dict)
        and isinstance(claim.get("text"), str)
        and claim["text"].strip()
    ]


def anchor_claim_texts(record: dict[str, Any]) -> list[str]:
    return claim_texts(record.get("anchor_claims") or [])


def valid_claims(record: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for paper_index, paper in enumerate(record.get("similar_papers") or [], 1):
        title = " ".join(str(paper.get("title", "")).split())
        external_ids = paper.get("external_ids") or {}
        doi = " ".join(str(external_ids.get("DOI", "")).split())
        for claim_index, text in enumerate(claim_texts(paper.get("claims") or []), 1):
            claims.append(
                {
                    "claim_id": f"paper_{paper_index}_claim_{claim_index}",
                    "paper_rank": str(paper.get("rank", paper_index)),
                    "title": title,
                    "doi": doi,
                    "text": text,
                }
            )
    return claims


def build_prompt(record: dict[str, Any]) -> dict[str, str]:
    request = {
        "retracted_claims": anchor_claim_texts(record),
        "valid_claims": valid_claims(record),
    }
    return {
        "system_prompt": SCORING_SYSTEM_PROMPT,
        "user_prompt": json.dumps(request, ensure_ascii=False, indent=2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate prompts for scoring parallelism between retracted and valid claims."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        source = json.load(input_file)

    records = source.get("records", []) if isinstance(source, dict) else source
    output_records = []
    generated = 0
    for record in records:
        output_record = dict(record)
        if anchor_claim_texts(record) and valid_claims(record):
            output_record["parallel_claim_prompt"] = build_prompt(record)
            output_record["parallel_claim_prompt_generation"] = {"status": "success"}
            generated += 1
        else:
            output_record["parallel_claim_prompt_generation"] = {
                "status": "skipped",
                "reason": "missing retracted anchor claims or valid claims",
            }
        output_records.append(output_record)

    output = dict(source) if isinstance(source, dict) else {}
    output["records"] = output_records
    output["parallel_claim_prompt_statistics"] = {
        "input_records": len(records),
        "prompts_generated": generated,
        "records_skipped": len(records) - generated,
        "model_calls_made": 0,
        "score_range": "0-5",
        "weights": {
            "same_problem_same_solution": 0.50,
            "same_problem_different_solution": 0.30,
            "same_solution_different_problem": 0.20,
        },
    }
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")

    print(f"Generated {generated} prompts in {args.output}; model calls made: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
