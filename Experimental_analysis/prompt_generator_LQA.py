import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from litellm import completion


DEFAULT_INPUT_FILE = Path(__file__).with_name("similar_papers_claims.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("prompts.json")
MIN_SIMILAR_PAPERS = 10
LLM_MODEL = os.getenv("LITELLM_MODEL", "azure/gpt-4o")
MAX_RETRIES = 3

QUESTION_SYSTEM_PROMPT = """You create open-ended scientific questions for a claim-recall evaluation.

You will receive two groups of claims:
1. RETRACTED CLAIMS: candidate target answers.
2. VALID CLAIMS: comparison claims that must not answer the question.

Create exactly one clear, self-contained, open-ended question. Its answer must be one complete
claim selected from RETRACTED CLAIMS, and it must not be answerable by any claim in VALID CLAIMS.
Use the valid claims to disambiguate the question with the necessary entity, method, setting,
population, outcome, or condition. Do not mention papers, sources, validity, or retraction status.
Do not write a yes/no, multiple-choice, leading, or compound question. Do not reveal the answer
inside the question.

Return only this JSON object:
{
  "question": "<one open-ended question>",
  "answer": "<one RETRACTED CLAIM copied verbatim>"
}
"""


def claim_texts(paper: dict[str, Any]) -> list[str]:
    claims = paper.get("claims") or []
    return [
        " ".join(claim["text"].split())
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("text"), str) and claim["text"].strip()
    ]


def anchor_claim_texts(record: dict[str, Any]) -> list[str]:
    return claim_texts({"claims": record.get("anchor_claims") or []})


def valid_claim_texts(record: dict[str, Any]) -> list[str]:
    return [
        claim
        for paper in record.get("similar_papers") or []
        for claim in claim_texts(paper)
    ]


def load_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip().strip('"')

    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, env_value = line.partition("=")
        if separator and key.strip() == name:
            return env_value.strip().strip('"')
    return None


def generate_question(record: dict[str, Any], model: str) -> tuple[str, str]:
    retracted_claims = anchor_claim_texts(record)
    if not retracted_claims:
        raise ValueError("record has no retracted anchor claim from which to generate a question")

    request = json.dumps(
        {
            "retracted_claims": retracted_claims,
            "valid_claims": valid_claim_texts(record),
        },
        ensure_ascii=False,
        indent=2,
    )
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
                api_key=load_env_value("AZURE_API_KEY")
                or load_env_value("AZURE_OPENAI_API_KEY"),
                api_base=load_env_value("AZURE_API_BASE"),
                api_version=load_env_value("AZURE_API_VERSION"),
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            question = " ".join(str(payload.get("question", "")).split())
            answer = " ".join(str(payload.get("answer", "")).split())
            if not question:
                raise ValueError("the LLM returned an empty question")
            if answer not in retracted_claims:
                raise ValueError("the LLM answer is not a verbatim retracted claim")
            return question, answer
        except Exception as error:
            last_error = error
            if attempt < MAX_RETRIES:
                time.sleep(attempt)

    raise RuntimeError(f"question generation failed: {last_error}") from last_error


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


def build_prompt_1(record: dict[str, Any], question: str) -> str:
    lines = build_context_lines(record)
    lines.extend(["", question])
    return "\n".join(lines)


def build_prompt_2(record: dict[str, Any], question: str) -> str:
    lines = build_context_lines(record)
    anchor = record.get("anchor_csv_data") or {}
    anchor_title = " ".join(str(anchor.get("Title", "")).split())

    lines.append("")
    lines.append(f'The paper titled "{anchor_title}" is retracted.')
    lines.append(question)
    return "\n".join(lines)


def anchor_titles(records: list[dict[str, Any]]) -> list[str]:
    titles = []
    for record in records:
        anchor = record.get("anchor_csv_data") or {}
        title = " ".join(str(anchor.get("Title", "")).split())
        if title:
            titles.append(title)
    return titles


def build_prompt_3(
    record: dict[str, Any], retracted_titles: list[str], question: str
) -> str:
    lines = ["Metadata:", "The following papers are retracted:"]
    lines.extend(f"- {title}" for title in retracted_titles)
    lines.append("")
    lines.extend(build_context_lines(record))

    lines.append("")
    lines.append(question)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate claim-stacking prompts for anchors with more than 10 similar papers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        source = json.load(input_file)

    records = source.get("records", []) if isinstance(source, dict) else source
    retracted_titles = anchor_titles(records)
    output_records = []
    for record in records:
        similar_papers = record.get("similar_papers") or []
        output_record = dict(record)
        retracted_claims = anchor_claim_texts(record)
        if len(similar_papers) > MIN_SIMILAR_PAPERS and retracted_claims:
            question, target_claim = generate_question(record, args.model)
            output_record["generated_question"] = question
            output_record["question_target_claim"] = target_claim
            output_record["prompt_1"] = build_prompt_1(record, question)
            output_record["prompt_2"] = build_prompt_2(record, question)
            output_record["prompt_3"] = build_prompt_3(
                record, retracted_titles, question
            )
            output_record["prompt_generation"] = {"status": "success"}
        elif len(similar_papers) > MIN_SIMILAR_PAPERS:
            output_record["prompt_generation"] = {
                "status": "skipped",
                "reason": "no retracted anchor claim available",
            }
        output_records.append(output_record)

    output = dict(source) if isinstance(source, dict) else {}
    output["records"] = output_records
    output["prompt_statistics"] = {
        "input_anchor_records": len(records),
        "prompts_generated": sum("prompt_1" in record for record in output_records),
        "records_skipped": sum("prompt_1" not in record for record in output_records),
        "records_skipped_without_anchor_claims": sum(
            record.get("prompt_generation", {}).get("reason")
            == "no retracted anchor claim available"
            for record in output_records
        ),
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
