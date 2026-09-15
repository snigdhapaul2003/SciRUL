"""Extract complete anchor-claim/answer samples from the experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_non_empty(value: Any) -> bool:
    """Return True for values containing meaningful data."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def has_non_empty_claim(claims: Any) -> bool:
    """Check that the claim list contains at least one usable claim."""
    if not isinstance(claims, list):
        return False
    return any(
        is_non_empty(claim.get("text")) if isinstance(claim, dict)
        else is_non_empty(claim)
        for claim in claims
    )


def extract_samples(data: dict[str, Any]) -> dict[str, Any]:
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError("The input field 'records' must be a list.")

    samples: list[dict[str, Any]] = []
    records_with_claims = 0
    records_without_claims = 0
    model_outputs_examined = 0
    records_without_complete_answers = 0

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            records_without_claims += 1
            continue

        anchor_claims = record.get("anchor_claims")
        if not has_non_empty_claim(anchor_claims):
            records_without_claims += 1
            continue

        records_with_claims += 1
        model_outputs = record.get("model_outputs", [])
        if not isinstance(model_outputs, list):
            continue

        metadata = record.get("anchor_csv_data", {})
        metadata = metadata if isinstance(metadata, dict) else {}

        answers: dict[str, Any] = {}
        answer_models: dict[str, Any] = {}
        answer_number = 1
        for output in model_outputs:
            model_outputs_examined += 1
            if not isinstance(output, dict):
                continue
            for source_key in ("answer_1", "answer_2", "answer_3"):
                answer = output.get(source_key)
                if is_non_empty(answer):
                    destination_key = f"answer_{answer_number}"
                    answers[destination_key] = answer
                    answer_models[destination_key] = output.get("model_name")
                    answer_number += 1

        # Three models, each with answer_1, answer_2, and answer_3, must
        # produce nine non-empty answers for the paper to be retained.
        if len(answers) != 9:
            records_without_complete_answers += 1
            continue

        samples.append(
            {
                "record_index": record_index,
                "record_id": metadata.get("Record ID"),
                "title": metadata.get("Title"),
                "anchor_claims": anchor_claims,
                "answers": answers,
                "answer_models": answer_models,
            }
        )

    return {
        "samples": samples,
        "statistics": {
            "total_input_records": len(records),
            "records_with_anchor_claims": records_with_claims,
            "records_without_anchor_claims": records_without_claims,
            "model_outputs_examined": model_outputs_examined,
            "samples_written": len(samples),
            "records_without_all_nine_answers": records_without_complete_answers,
            "total_answers_written": sum(
                len(sample["answers"]) for sample in samples
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all model answers into one sample per anchor paper."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("sample_input.json"),
        help="Input JSON path (default: sample_input.json beside this script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("claims_and_answers.json"),
        help="Output JSON path (default: claims_and_answers.json beside this script)",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError("The top-level JSON value must be an object.")

    result = extract_samples(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(json.dumps(result["statistics"], indent=2))
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
