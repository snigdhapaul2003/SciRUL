"""Combine positive and negative answer records from Sample_parent_data.

This is intentionally a data-only preparation step: existing judge prompts and
judge answers are ignored, and no LLM is called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ANSWER_KEYS = ("answer_1", "answer_2", "answer_3")


def non_empty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def records_from(data: Any, path: Path) -> list[Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")

    records = data.get("records", data.get("results"))
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a 'records' or 'results' list")
    return records


def combine_json_files(input_dir: Path, output_path: Path) -> dict[str, Any]:
    samples_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    files_read = 0
    records_read = 0
    records_without_answers = 0

    # Excluding the destination makes rerunning the script safe when the output
    # is placed inside the input directory.
    output_resolved = output_path.resolve()
    input_paths = sorted(
        path for path in input_dir.glob("*.json")
        if path.resolve() != output_resolved
    )

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as input_file:
            records = records_from(json.load(input_file), input_path)
        files_read += 1

        for source_record_index, record in enumerate(records):
            records_read += 1
            if not isinstance(record, dict):
                records_without_answers += 1
                continue

            metadata = record.get("anchor_csv_data")
            metadata = metadata if isinstance(metadata, dict) else {}
            record_id = metadata.get("Record ID", record.get("record_id"))
            is_negative = record.get("negative_sample") is True
            negative_sample_type = (
                record.get("negative_sample_type") if is_negative else None
            )
            # The tagged fallback prevents unrelated records with missing IDs
            # from being merged with one another. Positive and negative data
            # for the same ID are stored in separate sections of one sample.
            merge_key = (
                ("record_id", str(record_id))
                if non_empty(record_id)
                else ("source", f"{input_path.name}:{source_record_index}")
            )
            sample = samples_by_id.get(merge_key)
            if sample is None:
                sample = {
                    "record_index": record.get("record_index", source_record_index),
                    "record_id": record_id,
                    "title": metadata.get("Title", record.get("anchor_title")),
                    "anchor_claims": [],
                    "positive": {
                        "negative_sample": False,
                        "answers": {},
                        "answer_models": {},
                        "source_records": [],
                    },
                    "negative": {
                        "negative_sample": True,
                        "negative_sample_types": [],
                        "answers": {},
                        "answer_models": {},
                        "source_records": [],
                    },
                }
                samples_by_id[merge_key] = sample

            section = sample["negative" if is_negative else "positive"]
            if is_negative and non_empty(negative_sample_type):
                if negative_sample_type not in section["negative_sample_types"]:
                    section["negative_sample_types"].append(negative_sample_type)

            # Claim extraction and answer generation can live in different
            # source files. Merge claims before checking for answers so a
            # claim-only record can enrich an answer-only record with the same
            # record ID.
            existing_claims = {
                json.dumps(claim, ensure_ascii=False, sort_keys=True)
                for claim in sample["anchor_claims"]
            }
            for claim in record.get("anchor_claims") or []:
                serialized_claim = json.dumps(claim, ensure_ascii=False, sort_keys=True)
                if serialized_claim not in existing_claims:
                    sample["anchor_claims"].append(claim)
                    existing_claims.add(serialized_claim)

            section["source_records"].append(
                {"source_file": input_path.name, "source_record_index": source_record_index}
            )

            answers: dict[str, Any] = {}
            answer_models: dict[str, Any] = {}
            for model_output in record.get("model_outputs", []):
                if not isinstance(model_output, dict):
                    continue
                for source_answer_key in ANSWER_KEYS:
                    answer = model_output.get(source_answer_key)
                    if non_empty(answer):
                        destination_key = f"answer_{len(answers) + 1}"
                        answers[destination_key] = answer
                        answer_models[destination_key] = model_output.get("model_name")

            if not answers:
                records_without_answers += 1
                continue

            for local_answer_key, answer in answers.items():
                merged_answer_key = f"answer_{len(section['answers']) + 1}"
                section["answers"][merged_answer_key] = answer
                section["answer_models"][merged_answer_key] = answer_models[
                    local_answer_key
                ]
    # Claim-only records which never acquire answers are useful merger inputs
    # but are not trainable samples and must not appear in the final dataset.
    samples = [
        sample for sample in samples_by_id.values()
        if sample["positive"]["answers"] or sample["negative"]["answers"]
    ]

    return {
        "samples": samples,
        "statistics": {
            "input_files_read": files_read,
            "total_input_records": records_read,
            "samples_written": len(samples),
            "positive_samples_written": sum(
                bool(sample["positive"]["answers"]) for sample in samples
            ),
            "negative_samples_written": sum(
                bool(sample["negative"]["answers"]) for sample in samples
            ),
            "source_records_merged": sum(
                len(sample[section]["source_records"])
                for sample in samples
                for section in ("positive", "negative")
            ) - len(samples),
            "records_without_answers": records_without_answers,
            "total_answers_written": sum(
                len(sample[section]["answers"])
                for sample in samples
                for section in ("positive", "negative")
            ),
        },
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Combine all Sample_parent_data JSON records without LLM judging."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=script_dir / "Sample_parent_data",
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=script_dir / "claims_and_answers.json",
        help="Unified output JSON path",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    result = combine_json_files(args.input_dir, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(json.dumps(result["statistics"], indent=2))
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
