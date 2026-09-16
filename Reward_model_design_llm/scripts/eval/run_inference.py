"""Run the frozen retrieval, derivation, verification, and aggregation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from tqdm.auto import tqdm

from retracted_claims.core import CosineVerifier, aggregate, step_dict
from retracted_claims.data import sentences_with_offsets
from retracted_claims.llama import LlamaDeriver
from retracted_claims.retrieval import ScientificIndex


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Paragraph-only inference JSON")
    parser.add_argument("checkpoint", help="Frozen base Llama checkpoint")
    parser.add_argument("output", help="Prediction JSON destination")
    parser.add_argument("--index", required=True, help="Frozen claim-index NPZ")
    parser.add_argument("--adapter", help="Optional fine-tuned LoRA adapter")
    parser.add_argument("--conformal", help="Optional calibration JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.55,
        help="Minimum claim/span SciNCL similarity",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Fallback self-consistency threshold when no calibration file is given",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args(argv)


def load_paragraphs(path: str | Path) -> list[dict[str, str]]:
    """Read the deployment input and reject accidental gold/claim fields."""
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    paragraphs = root.get("paragraphs") if isinstance(root, dict) else None
    if not isinstance(paragraphs, list):
        raise ValueError("inference input must contain a 'paragraphs' list")

    output = []
    for row in paragraphs:
        if not isinstance(row, dict) or set(row) != {"paragraph_id", "paragraph"}:
            raise ValueError(
                "each inference row must contain only paragraph_id and paragraph"
            )
        output.append(
            {
                "paragraph_id": str(row["paragraph_id"]),
                "paragraph": str(row["paragraph"]),
            }
        )
    return output


def load_confidence_threshold(args) -> float:
    if not args.conformal:
        return args.confidence_threshold
    calibration = json.loads(Path(args.conformal).read_text(encoding="utf-8"))
    threshold = calibration.get("step_confidence_threshold")
    if not isinstance(threshold, (int, float)):
        raise ValueError("calibration JSON lacks step_confidence_threshold")
    return float(threshold)


def paragraph_verdict(sentence_verdicts: list[str]) -> str:
    if "used" in sentence_verdicts:
        return "used"
    if "abstain" in sentence_verdicts:
        return "abstain"
    return "not_used"


def main(argv=None):
    args = parse_arguments(argv)
    if args.top_k < 1:
        raise ValueError("top-k must be at least one")
    if args.samples < 1:
        raise ValueError("samples must be at least one")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be at least one")
    if not 0.0 <= args.semantic_threshold <= 1.0:
        raise ValueError("semantic-threshold must be between zero and one")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be between zero and one")
    torch.set_grad_enabled(False)
    assert not torch.is_grad_enabled()

    paragraphs = load_paragraphs(args.input)
    index = ScientificIndex.load(args.index)
    claims_by_id = {claim.claim_id: claim for claim in index.claims}
    confidence_threshold = load_confidence_threshold(args)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("calibrated confidence threshold must be between zero and one")

    deriver = LlamaDeriver(
        args.checkpoint,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    verifier = CosineVerifier(
        threshold=args.semantic_threshold,
        encoder=index,
    )
    assert not deriver.model.training
    assert all(not parameter.requires_grad for parameter in deriver.model.parameters())

    total_sentences = sum(
        len(sentences_with_offsets(row["paragraph"])) for row in paragraphs
    )
    progress = tqdm(
        total=total_sentences,
        desc="Inference",
        unit="sentence",
        dynamic_ncols=True,
    )

    predictions = []
    parse_failures = 0
    generated_sequences = 0
    for row in paragraphs:
        paragraph_id = row["paragraph_id"]
        paragraph = row["paragraph"]
        sentence_outputs = []
        verdicts_by_claim: dict[str, list[str]] = defaultdict(list)
        steps_by_claim: dict[str, list[dict]] = defaultdict(list)

        for start, end, sentence in sentences_with_offsets(paragraph):
            retrieved_claims = index.retrieve(sentence, args.top_k)
            retrieved_paper_ids = []
            for claim, _score in retrieved_claims:
                if claim.paper_id not in retrieved_paper_ids:
                    retrieved_paper_ids.append(claim.paper_id)

            sentence_claim_outputs = []
            for paper_id in retrieved_paper_ids:
                sibling_claims = [
                    claim for claim in index.claims if claim.paper_id == paper_id
                ]
                if len(sibling_claims) != 5:
                    raise AssertionError(
                        f"paper {paper_id} has {len(sibling_claims)} claims, expected 5"
                    )

                generated_steps, failures = deriver.derive_batch(
                    [(claim.text, sentence) for claim in sibling_claims],
                    args.samples,
                )
                parse_failures += failures
                generated_sequences += len(sibling_claims) * args.samples
                verifier.verify_batch(
                    [
                        (claim.text, sentence, steps, start)
                        for claim, steps in zip(sibling_claims, generated_steps)
                    ]
                )

                for claim, steps in zip(sibling_claims, generated_steps):
                    verdict = aggregate(steps, confidence_threshold)
                    serialized_steps = [step_dict(step, paragraph) for step in steps]
                    accepted_steps = [
                        step_dict(step, paragraph)
                        for step in steps
                        if step.status == "verified_positive"
                        and step.score >= confidence_threshold
                    ]
                    verdicts_by_claim[claim.claim_id].append(verdict)
                    # Paragraph-level evidence contains only steps accepted by
                    # both deterministic verification and conformal control.
                    steps_by_claim[claim.claim_id].extend(accepted_steps)
                    sentence_claim_outputs.append(
                        {
                            "claim_id": claim.claim_id,
                            "steps": serialized_steps,
                            "verdict": verdict,
                        }
                    )

            sentence_outputs.append(
                {
                    "start": start,
                    "end": end,
                    "sentence": sentence,
                    "claims": sentence_claim_outputs,
                }
            )
            progress.update(1)
            progress.set_postfix(
                paragraph=paragraph_id,
                generations=generated_sequences,
                failures=parse_failures,
            )

        # Claims absent from retrieval receive an explicit not_used verdict.
        paragraph_claims = []
        for claim_id in claims_by_id:
            sentence_verdicts = verdicts_by_claim.get(claim_id, [])
            paragraph_claims.append(
                {
                    "claim_id": claim_id,
                    "verdict": paragraph_verdict(sentence_verdicts),
                    "steps": steps_by_claim.get(claim_id, []),
                }
            )

        predictions.append(
            {
                "paragraph_id": paragraph_id,
                "paragraph": paragraph,
                "sentences": sentence_outputs,
                "claims": paragraph_claims,
            }
        )

    progress.close()
    report = {
        "pipeline_input": "paragraphs_only",
        "claim_index": str(Path(args.index).resolve()),
        "conformal_file": (
            str(Path(args.conformal).resolve()) if args.conformal else None
        ),
        "step_confidence_threshold": confidence_threshold,
        "semantic_threshold": args.semantic_threshold,
        "parse_failures": parse_failures,
        "llama_generated_sequences": generated_sequences,
        "predictions": predictions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
