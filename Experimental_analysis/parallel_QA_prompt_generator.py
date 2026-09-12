import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from litellm import completion


DEFAULT_PROMPTS_FILE = Path(__file__).with_name("parallel_claim_prompts.json")
DEFAULT_SCORES_FILE = Path(__file__).with_name("parallel_claim_scores.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("parallel_QA_prompts.json")
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "azure/gpt-5.5")
DEFAULT_MAX_RETRIES = 3
MIN_OVERALL_SCORE = 2.5

CATEGORY_NAMES = {
    "same_problem_same_solution_score": "same_problem_same_solution",
    "same_problem_different_solution_score": "same_problem_different_solution",
    "same_solution_different_problem_score": "same_solution_different_problem",
}


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
    required = {
        "api_key": api_key,
        "api_base": values.get("AZURE_API_BASE"),
        "api_version": values.get("AZURE_API_VERSION"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Azure settings: {', '.join(missing)}")
    return {name: value for name, value in required.items() if value}


def parse_json_response(response: Any) -> dict[str, Any]:
    content = response.choices[0].message.content
    if not content:
        raise ValueError("the model returned an empty response")
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
        raise ValueError("the model response must contain a question string")
    return payload


def dominant_categories(score: dict[str, Any]) -> list[str]:
    category_scores = {
        name: float(score.get(field, 0))
        for field, name in CATEGORY_NAMES.items()
    }
    highest = max(category_scores.values())
    return [name for name, value in category_scores.items() if value == highest]


def candidate_claims(record: dict[str, Any], paper_rank: str) -> list[dict[str, Any]]:
    prompt = record.get("parallel_claim_prompt") or {}
    user_prompt = prompt.get("user_prompt", "")
    if not isinstance(user_prompt, str):
        return []
    try:
        payload = json.loads(user_prompt)
    except json.JSONDecodeError:
        return []
    return [
        claim
        for claim in payload.get("valid_claims", [])
        if str(claim.get("paper_rank")) == str(paper_rank)
    ]


def get_record_claim_payload(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record.get("parallel_claim_prompt") or {}
    user_prompt = prompt.get("user_prompt", "")
    if not isinstance(user_prompt, str):
        raise ValueError("parallel_claim_prompt.user_prompt is missing")
    try:
        payload = json.loads(user_prompt)
    except json.JSONDecodeError as error:
        raise ValueError("parallel_claim_prompt.user_prompt is not valid JSON") from error
    return payload


def all_valid_papers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    papers: dict[tuple[str, str], dict[str, Any]] = {}
    for claim in payload.get("valid_claims", []):
        rank = str(claim.get("paper_rank", ""))
        title = claim.get("title", "")
        key = (rank, title)
        paper = papers.setdefault(
            key,
            {"title": title, "claims": []},
        )
        paper["claims"].append({"text": claim.get("text", "")})
    return list(papers.values())


def neutral_claim_context(payload: dict[str, Any], anchor_title: str) -> list[dict[str, Any]]:
    return [
        {"title": anchor_title, "claims": [{"text": text} for text in payload.get("retracted_claims", [])]}
    ] + all_valid_papers(payload)


def build_question_prompt(
    record: dict[str, Any],
    score: dict[str, Any],
    claims: list[dict[str, Any]],
) -> str:
    payload = get_record_claim_payload(record)
    anchor_title = (record.get("anchor_csv_data") or {}).get("Title", "")
    retracted_claims = payload.get("retracted_claims", [])
    category = dominant_categories(score)
    category_instruction = {
        "same_problem_same_solution": (
            "The problem and solution are shared. Ask about the shared scientific problem "
            "and how the candidate claim uses the same or closely related solution."
        ),
        "same_problem_different_solution": (
            "The problem is shared but the solution differs. Ask about the shared problem "
            "and how the candidate claim solves it differently."
        ),
        "same_solution_different_problem": (
            "The solution is shared but the problem differs. Ask which method is shared and "
            "how the candidate applies it to a different problem."
        ),
    }
    if len(category) == 1:
        instruction = category_instruction[category[0]]
    else:
        instruction = (
            "The highest categories are tied. Ask a balanced question that identifies the "
            "shared problem and/or solution and clearly distinguishes the remaining difference."
        )

    context = {"retracted_claims": retracted_claims, "candidate_claims": claims}
    return f"""Generate one standalone open-ended academic question about the shared scientific topic.

{instruction}

The question will be shown to a respondent without the evidence used to design it. It must be
self-contained and answerable from either body of scientific content independently. Ask a
natural open question about the shared domain, mechanism, task, outcome, or practical meaning.
Where the selected category requires a distinction, phrase the question as a general domain
question that naturally allows different approaches or applications to be explained.

Strictly do not mention: claims, claim sets, papers, articles, studies, sources, authors,
titles, ranks, DOIs, retracted work, candidate work, comparisons, "both", or "according to".
Do not ask for a numerical score. Do not ask the respondent to compare two methods or texts.
Return JSON only with a concise question and a one-sentence focus.

EVIDENCE FOR QUESTION DESIGN:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return:
{{
    "question": "one standalone open-ended domain question",
  "focus": "what the question tests"
}}
"""


def build_final_prompt(
    record: dict[str, Any],
    generated_question: dict[str, Any],
    retraction_metadata: str = "",
) -> str:
    payload = get_record_claim_payload(record)
    context = {
        "papers": neutral_claim_context(payload, (record.get("anchor_csv_data") or {}).get("Title", "")),
    }
    prompt = f"""Answer the question below using the complete set of paper titles and claims provided.

Give a neutral, evidence-based answer grounded only in the supplied titles and claims.
Do not mention paper ranks, claim ranks, claim IDs, internal record IDs, or the origin/status
of any paper. Do not invent facts absent from the
supplied material.

QUESTION TO ANSWER:
{generated_question.get("question", "")}

PAPER TITLES AND CLAIMS:
{json.dumps(context, ensure_ascii=False, indent=2)}

Write a concise but sufficiently detailed answer grounded only in the claims above.
"""
    if retraction_metadata:
        prompt += f"\n{retraction_metadata.strip()}\n"
    return prompt


def unique_retracted_titles(records: list[dict[str, Any]]) -> list[str]:
    """Collect unique anchor titles in source order."""
    titles: list[str] = []
    seen: set[str] = set()
    for record in records:
        title = " ".join(
            str((record.get("anchor_csv_data") or {}).get("Title", "")).split()
        )
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def build_final_prompts(
    record: dict[str, Any],
    generated_question: dict[str, Any],
    all_retracted_titles: list[str],
) -> tuple[str, str, str]:
    """Build neutral, single-title, and all-title metadata variants."""
    anchor_title = " ".join(
        str((record.get("anchor_csv_data") or {}).get("Title", "")).split()
    )
    single_title_metadata = (
        f'RETRACTED PAPER METADATA: The paper titled "{anchor_title}" is retracted.'
    )
    all_titles_metadata = (
        "RETRACTED PAPER METADATA:\n"
        "The following paper titles are retracted:\n"
        + json.dumps(all_retracted_titles, ensure_ascii=False, indent=2)
    )
    return (
        build_final_prompt(record, generated_question),
        build_final_prompt(record, generated_question, single_title_metadata),
        build_final_prompt(record, generated_question, all_titles_metadata),
    )


def generate_question(
    final_prompt: str,
    model: str,
    azure_config: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    options: dict[str, Any] = {"response_format": {"type": "json_object"}}
    if not model.lower().split("/")[-1].startswith("gpt-5"):
        options["temperature"] = 0

    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                **options,
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
        description="Generate paper-level QA prompts from parallel claim scores."
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.prompts.open("r", encoding="utf-8") as prompts_file:
        prompt_source = json.load(prompts_file)
    with args.scores.open("r", encoding="utf-8") as scores_file:
        score_source = json.load(scores_file)

    records = prompt_source.get("records", [])
    score_records = score_source.get("results", [])
    retracted_titles = unique_retracted_titles(records)
    azure_config = load_env_values(args.prompts.with_name(".env"))
    output_records = []
    skipped = {"rank_1": 0, "below_threshold": 0, "missing_claims": 0}

    for score_record in score_records:
        record_index = score_record.get("record_index")
        if not isinstance(record_index, int) or record_index >= len(records):
            continue
        record = records[record_index]
        paper_scores = (score_record.get("parallel_paper_scores") or {}).get("paper_scores", [])
        for score in paper_scores:
            paper_rank = str(score.get("paper_rank", ""))
            overall_score = float(score.get("overall_weighted_score", 0))
            if paper_rank == "1":
                skipped["rank_1"] += 1
                continue
            if overall_score < MIN_OVERALL_SCORE:
                skipped["below_threshold"] += 1
                continue

            claims = candidate_claims(record, paper_rank)
            if not claims:
                skipped["missing_claims"] += 1
                continue
            if args.limit is not None and len(output_records) >= args.limit:
                break
            anchor_title = (record.get("anchor_csv_data") or {}).get("Title", "")
            candidate_title = score.get("title", "")
            question_prompt = build_question_prompt(record, score, claims)
            result = {
                "record_index": record_index,
                "record_id": (record.get("anchor_csv_data") or {}).get("Record ID"),
                "anchor_title": anchor_title,
                "candidate_paper_rank": paper_rank,
                "candidate_title": candidate_title,
                "overall_weighted_score": overall_score,
                "highest_categories": dominant_categories(score),
                "question_generation_prompt": question_prompt,
            }
            try:
                result["generated_question"] = generate_question(
                    question_prompt, args.model, azure_config, args.max_retries
                )
                final_prompts = build_final_prompts(
                    record, result["generated_question"], retracted_titles
                )
                for prompt_number, final_prompt in enumerate(final_prompts, start=1):
                    result[f"final_prompt_{prompt_number}"] = final_prompt
                result["status"] = "success"
            except Exception as error:
                result["status"] = "error"
                result["error"] = f"{type(error).__name__}: {error}"
            output_records.append(result)
            print(f"Generated {result['status']}: {candidate_title}")
        if args.limit is not None and len(output_records) >= args.limit:
            break

    output = {
        "model": args.model,
        "score_threshold": MIN_OVERALL_SCORE,
        "excluded_rank": 1,
        "results": output_records,
        "statistics": {
            "prompts_generated": len(output_records),
            "successful_questions": sum(item["status"] == "success" for item in output_records),
            "failed_questions": sum(item["status"] == "error" for item in output_records),
            "skipped": skipped,
        },
    }
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")
    print(f"Saved {len(output_records)} QA prompt records to {args.output}")
    return 0 if output["statistics"]["failed_questions"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
