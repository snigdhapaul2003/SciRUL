"""Create valid-claim-only negative summary samples.

The source summary prompts include the retracted/anchor claim followed by
similar papers.  For the negative ablation we keep the same summary topic, but
rebuild the prompt using only valid similar-paper claims.  The anchor claims are
kept as metadata so the downstream judge can still verify that the answer does
not use them.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


PROMPT_MARKER = "Using the above context, generate a summary for the following topic:"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "Sample_parent_data"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "negative_summary_answers.json"
DEFAULT_MODELS = (
    "olmo_3_7B_Instruct",
    "llama_3_8B_Instruct",
    "gpt-5.5",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def claim_to_text(claim: Any) -> str:
    if isinstance(claim, dict):
        for key in ("text", "claim", "sentence"):
            value = claim.get(key)
            if value:
                return str(value).strip()
    return str(claim or "").strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_topic(record: dict[str, Any]) -> str | None:
    for prompt_name in ("prompt_1", "prompt_2", "prompt_3"):
        prompt = record.get(prompt_name)
        if not isinstance(prompt, str):
            continue
        marker_index = prompt.rfind(PROMPT_MARKER)
        if marker_index >= 0:
            topic = prompt[marker_index + len(PROMPT_MARKER) :].strip()
            if topic:
                return topic
    return None


def parse_rank(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_valid_similar_papers(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return papers used in the negative prompt and papers explicitly excluded."""
    anchor_title = normalize_text(record.get("anchor_csv_data", {}).get("Title"))
    valid_papers: list[dict[str, Any]] = []
    excluded_papers: list[dict[str, Any]] = []

    for paper in record.get("similar_papers", []) or []:
        rank = parse_rank(paper.get("rank"))
        title = normalize_text(paper.get("title"))
        claims = [claim_to_text(claim) for claim in paper.get("claims", []) or [] if claim_to_text(claim)]
        is_rank_one = rank is not None and rank <= 1
        is_anchor_title = bool(anchor_title and title == anchor_title)
        is_retracted_title = title.startswith("retracted:")

        # The first source block in the original prompt is the anchor claim.
        # The rank-1 similar paper is the retracted paper match, so it also
        # stays out of the negative prompt.
        if is_rank_one or is_anchor_title or is_retracted_title or not claims:
            excluded_papers.append(
                {
                    "rank": rank,
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "reason": "invalid_or_retracted_source" if claims else "no_claims",
                }
            )
            continue

        paper_copy = copy.deepcopy(paper)
        paper_copy["claims"] = claims
        valid_papers.append(paper_copy)

    return valid_papers, excluded_papers


def build_context(valid_papers: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for paper in valid_papers:
        title = str(paper.get("title") or "").strip()
        for claim in paper.get("claims", []) or []:
            claim_text = claim_to_text(claim)
            if not claim_text:
                continue
            blocks.append(f"Title: {title}\nClaim: {claim_text}")
    return "\n".join(blocks)


def get_forbidden_claims(record: dict[str, Any]) -> list[str]:
    forbidden_claims: list[str] = []
    forbidden_claims.extend(claim_to_text(claim) for claim in record.get("anchor_claims", []) or [])
    for paper in record.get("similar_papers", []) or []:
        rank = parse_rank(paper.get("rank"))
        title = normalize_text(paper.get("title"))
        if (rank is not None and rank <= 1) or title.startswith("retracted:"):
            forbidden_claims.extend(claim_to_text(claim) for claim in paper.get("claims", []) or [])
    return forbidden_claims


def remove_forbidden_claims_from_valid_papers(
    valid_papers: list[dict[str, Any]],
    forbidden_claims: list[str],
) -> tuple[list[dict[str, Any]], int]:
    forbidden = {normalize_text(claim) for claim in forbidden_claims if normalize_text(claim)}
    filtered_papers: list[dict[str, Any]] = []
    removed_claim_count = 0

    for paper in valid_papers:
        paper_copy = copy.deepcopy(paper)
        kept_claims = []
        for claim in paper.get("claims", []) or []:
            claim_text = claim_to_text(claim)
            if normalize_text(claim_text) in forbidden:
                removed_claim_count += 1
                continue
            kept_claims.append(claim_text)
        if kept_claims:
            paper_copy["claims"] = kept_claims
            filtered_papers.append(paper_copy)

    return filtered_papers, removed_claim_count


def contains_forbidden_claim(prompt: str, forbidden_claims: list[str]) -> bool:
    normalized_prompt = normalize_text(prompt)
    for claim in forbidden_claims:
        normalized_claim = normalize_text(claim)
        if normalized_claim and normalized_claim in normalized_prompt:
            return True
    return False


def make_negative_record(
    record: dict[str, Any],
    *,
    source_file: str,
    source_record_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    topic = extract_topic(record)
    if not topic:
        return None, "missing_summary_topic"

    valid_papers, excluded_papers = get_valid_similar_papers(record)
    forbidden_claims = get_forbidden_claims(record)
    valid_papers, removed_duplicate_forbidden_claims = remove_forbidden_claims_from_valid_papers(
        valid_papers,
        forbidden_claims,
    )
    context = build_context(valid_papers)
    if not context:
        return None, "no_valid_claim_context"

    negative_prompt = f"{context}\n\n{PROMPT_MARKER} {topic}"

    if contains_forbidden_claim(negative_prompt, forbidden_claims):
        return None, "forbidden_claim_leaked_into_prompt"

    negative_record = copy.deepcopy(record)
    negative_record["source_summary_file"] = source_file
    negative_record["source_summary_record_index"] = source_record_index
    negative_record["negative_sample"] = True
    negative_record["negative_sample_type"] = "summary_valid_claims_only"
    negative_record["excluded_from_negative_prompt"] = {
        "anchor_title": record.get("anchor_csv_data", {}).get("Title"),
        "anchor_claim_count": len(record.get("anchor_claims", []) or []),
        "similar_papers": excluded_papers,
    }
    negative_record["valid_prompt_paper_count"] = len(valid_papers)
    negative_record["valid_prompt_claim_count"] = sum(
        len([claim for claim in paper.get("claims", []) or [] if claim_to_text(claim)])
        for paper in valid_papers
    )
    negative_record["removed_duplicate_forbidden_claim_count"] = removed_duplicate_forbidden_claims
    for prompt_number in range(1, 4):
        negative_record[f"prompt_{prompt_number}"] = negative_prompt
    negative_record["model_outputs"] = []
    return negative_record, None


def build_negative_document(input_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    source_statistics: dict[str, dict[str, int]] = {}

    for source_path in sorted(input_dir.glob("judge_answers_summary_*.json")):
        document = read_json(source_path)
        source_records = document.get("records", [])
        if not isinstance(source_records, list):
            skipped["invalid_records_list"] = skipped.get("invalid_records_list", 0) + 1
            continue

        built_for_file = 0
        for source_record_index, record in enumerate(source_records):
            if not isinstance(record, dict):
                skipped["non_dict_record"] = skipped.get("non_dict_record", 0) + 1
                continue
            if not all(isinstance(record.get(f"prompt_{i}"), str) for i in range(1, 4)):
                skipped["missing_prompts"] = skipped.get("missing_prompts", 0) + 1
                continue

            negative_record, skip_reason = make_negative_record(
                record,
                source_file=source_path.name,
                source_record_index=source_record_index,
            )
            if negative_record is None:
                skipped[skip_reason or "unknown"] = skipped.get(skip_reason or "unknown", 0) + 1
                continue
            records.append(negative_record)
            built_for_file += 1

        source_statistics[source_path.name] = {
            "source_records": len(source_records),
            "negative_records": built_for_file,
        }

    return {
        "source_files": sorted(path.name for path in input_dir.glob("judge_answers_summary_*.json")),
        "negative_sample_type": "summary_valid_claims_only",
        "prompt_policy": {
            "include": "rank >= 2 similar-paper claims",
            "exclude": "anchor claims and rank-1/anchor-title similar paper claims",
            "prompt_1_to_3": "same valid-claims-only summary prompt, generated three times for answer slots",
        },
        "records": records,
        "statistics": {
            "negative_records": len(records),
            "negative_prompt_count": len(records) * 3,
            "source_files": source_statistics,
            "skipped": skipped,
        },
    }


def generate_answers(
    output_path: Path,
    *,
    models: tuple[str, ...],
    max_retries: int,
    temperature: float,
    max_new_tokens: int,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(project_root / ".env")

    model_runner_dir = project_root / "Experimental_analysis"
    sys.path.insert(0, str(model_runner_dir))
    from model_run import run as run_model_generation

    run_model_generation(
        output_path,
        output_path,
        models,
        max_retries,
        temperature,
        max_new_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build negative summary samples with only valid claims in the prompt."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Only write the negative prompt JSON; do not generate model answers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rebuilding prompts when the output JSON already exists and generate missing answers.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.output.exists():
        document = read_json(args.output)
    else:
        document = build_negative_document(args.input_dir)
        write_json(args.output, document)

    stats = document.get("statistics", {})
    print(f"Saved negative prompts to {args.output}")
    print(f"Negative records: {stats.get('negative_records', 0)}")
    print(f"Negative prompts: {stats.get('negative_prompt_count', 0)}")
    skipped = stats.get("skipped", {})
    if skipped:
        print(f"Skipped: {skipped}")

    if not args.prompts_only:
        generate_answers(
            args.output,
            models=tuple(args.models),
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
