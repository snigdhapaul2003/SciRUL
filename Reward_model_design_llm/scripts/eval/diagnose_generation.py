"""Explain strict-JSON generation failures on labeled test examples."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from retracted_claims.data import load_dataset, sentences_with_offsets, split_answers
from retracted_claims.llama import LlamaDeriver


def inspect_response(raw: str, sentence: str) -> tuple[str, str, list[dict]]:
    """Return (status, explanation, normalized steps) without hiding raw output."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return "malformed_json", f"{error.msg} at character {error.pos}", []
    if not isinstance(value, dict):
        return "wrong_schema", "top-level JSON value is not an object", []
    if set(value) != {"steps"}:
        return "wrong_schema", "top-level JSON must contain only 'steps'", []
    steps = value.get("steps")
    if not isinstance(steps, list):
        return "wrong_schema", "missing 'steps' list", []
    normalized = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return "wrong_schema", f"steps[{index}] is not an object", []
        if not isinstance(step.get("span"), str):
            return "wrong_schema", f"steps[{index}].span is not a string", []
        if set(step) != {"span"}:
            return "wrong_schema", f"steps[{index}] must contain only 'span'", []
        span = step["span"]
        if not span:
            return "wrong_schema", "use an empty steps list for no evidence", []
        if span not in sentence:
            return "non_verbatim_span", f"span is not an exact sentence substring: {span!r}", []
        normalized.append({"span": span})
    return "valid", "strict JSON and all spans are grounded", normalized


def candidates(input_path: Path, seed: int):
    for example in split_answers(load_dataset(input_path), seed)["test"]:
        for start, end, sentence in sentences_with_offsets(example.paragraph):
            for claim in example.claims:
                gold = [
                    span.text for span in example.spans
                    if claim.claim_number in span.claim_numbers
                    and span.start >= start and span.end <= end
                ]
                yield {
                    "paragraph_id": example.paragraph_id,
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "sentence": sentence,
                    "gold_used": bool(gold),
                    "gold_spans": gold,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("--adapter")
    parser.add_argument("--positive-examples", type=int, default=3)
    parser.add_argument("--negative-examples", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = LlamaDeriver(
        args.checkpoint,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    all_candidates = list(candidates(args.input, args.seed))
    selected = (
        [x for x in all_candidates if x["gold_used"]][:args.positive_examples]
        + [x for x in all_candidates if not x["gold_used"]][:args.negative_examples]
    )
    report = []
    counts = Counter()
    for example_number, item in enumerate(selected, 1):
        for sample_number in range(1, args.samples + 1):
            raw = model._once(item["claim"], item["sentence"], args.samples > 1)
            status, explanation, steps = inspect_response(raw, item["sentence"])
            counts[status] += 1
            row = {
                **item,
                "sample": sample_number,
                "status": status,
                "explanation": explanation,
                "parsed_steps": steps,
                "raw_output": raw,
            }
            report.append(row)
            print(f"\n--- Example {example_number}, sample {sample_number} ---")
            print(f"Gold used: {item['gold_used']} | Status: {status}")
            print(f"Reason: {explanation}")
            print(f"Claim: {item['claim']}")
            print(f"Sentence: {item['sentence']}")
            print("Raw model output:")
            print(repr(raw))

    result = {"summary": dict(counts), "generations": report}
    print("\n=== Summary ===")
    print(json.dumps(result["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved diagnostic report to {args.output}")


if __name__ == "__main__":
    main()
