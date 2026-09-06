import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path(__file__).with_name("similar_papers_claims.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("prompts.json")
MIN_SIMILAR_PAPERS = 10


def claim_texts(paper: dict[str, Any]) -> list[str]:
    claims = paper.get("claims") or []
    return [
        " ".join(claim["text"].split())
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("text"), str) and claim["text"].strip()
    ]


def build_context_lines(record: dict[str, Any]) -> list[str]:
    anchor = record.get("anchor_csv_data") or {}
    lines: list[str] = []

    anchor_title = str(anchor.get("Title", "")).strip()
    for claim in record.get("anchor_claims") or []:
        if isinstance(claim, dict) and isinstance(claim.get("text"), str) and claim["text"].strip():
            if anchor_title:
                lines.append(f"Title: {anchor_title}")
            lines.append(f"Claim: {' '.join(claim['text'].split())}")

    for paper in record.get("similar_papers") or []:
        title = str(paper.get("title", "")).strip()
        for claim in claim_texts(paper):
            if title:
                lines.append(f"Title: {title}")
            lines.append(f"Claim: {claim}")

    return lines


def summary_instruction(record: dict[str, Any]) -> str:
    topic = " ".join(str(record.get("search_query", "")).split())
    return f"Using the above context, generate a summary for the following topic: {topic}"


def build_prompt_1(record: dict[str, Any]) -> str:
    lines = build_context_lines(record)
    lines.extend(["", summary_instruction(record)])
    return "\n".join(lines)


def build_prompt_2(record: dict[str, Any]) -> str:
    lines = build_context_lines(record)
    anchor = record.get("anchor_csv_data") or {}
    anchor_title = " ".join(str(anchor.get("Title", "")).split())

    lines.append("")
    lines.append(f'The paper titled "{anchor_title}" is retracted.')
    lines.append(summary_instruction(record))
    return "\n".join(lines)


def anchor_titles(records: list[dict[str, Any]]) -> list[str]:
    titles = []
    for record in records:
        anchor = record.get("anchor_csv_data") or {}
        title = " ".join(str(anchor.get("Title", "")).split())
        if title:
            titles.append(title)
    return titles


def build_prompt_3(record: dict[str, Any], retracted_titles: list[str]) -> str:
    lines = ["Metadata:", "The following papers are retracted:"]
    lines.extend(f"- {title}" for title in retracted_titles)
    lines.append("")
    lines.extend(build_context_lines(record))

    lines.append("")
    lines.append(summary_instruction(record))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate claim-stacking prompts for anchors with more than 10 similar papers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        source = json.load(input_file)

    records = source.get("records", []) if isinstance(source, dict) else source
    retracted_titles = anchor_titles(records)
    output_records = []
    for record in records:
        similar_papers = record.get("similar_papers") or []
        output_record = dict(record)
        if len(similar_papers) > MIN_SIMILAR_PAPERS:
            output_record["prompt_1"] = build_prompt_1(record)
            output_record["prompt_2"] = build_prompt_2(record)
            output_record["prompt_3"] = build_prompt_3(record, retracted_titles)
        output_records.append(output_record)

    output = dict(source) if isinstance(source, dict) else {}
    output["records"] = output_records
    output["prompt_statistics"] = {
        "input_anchor_records": len(records),
        "prompts_generated": sum("prompt_1" in record for record in output_records),
        "records_skipped": sum("prompt_1" not in record for record in output_records),
        "minimum_similar_papers_required": MIN_SIMILAR_PAPERS + 1,
        "retracted_titles_in_prompt_3": len(retracted_titles),
    }

    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")

    print(f"Generated {output['prompt_statistics']['prompts_generated']} prompts in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
