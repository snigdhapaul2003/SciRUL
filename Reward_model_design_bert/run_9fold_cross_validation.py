"""Nine-fold cross-validation, holding out answer_1 through answer_9 in turn."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluate_character_overlap import evaluate
from retracted_claim_pipeline import (
    CROSS_ENCODER,
    CandidateNarrower,
    SentenceSegmenter,
    JointCollator,
    JointDebertaTagger,
    PairExample,
    ScientificEncoder,
    SplitConformalBinary,
    asdict,
    load_pairs,
    prediction_record,
)


METRICS_TO_AVERAGE = (
    "micro_character_precision",
    "micro_character_recall",
    "micro_character_f1",
    "micro_character_iou",
    "macro_character_precision",
    "macro_character_recall",
    "macro_character_f1",
    "macro_character_iou",
    "exact_character_set_match_rate",
)
PIPELINE_VERSION = 4


def answer_number(example: PairExample) -> int:
    try:
        number = int(example.paragraph_id.rsplit(":answer_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot find answer number in {example.paragraph_id!r}") from error
    if number not in range(1, 10):
        raise ValueError(f"Answer number must be 1..9, got {number}")
    return number


def fold_split(
    examples: Sequence[PairExample], test_answer: int
) -> tuple[list[PairExample], list[PairExample], list[PairExample], int]:
    """Use one answer for test and a two-class answer for calibration."""
    calibration_answer = None
    for offset in range(1, 9):
        candidate = (test_answer - 1 + offset) % 9 + 1
        labels = {
            example.used for example in examples
            if answer_number(example) == candidate
        }
        if labels == {0, 1}:
            calibration_answer = candidate
            break
    if calibration_answer is None:
        raise ValueError(
            f"No non-test answer position contains both calibration classes "
            f"for fold {test_answer}"
        )
    train, calibration, test = [], [], []
    for example in examples:
        number = answer_number(example)
        if number == test_answer:
            test.append(example)
        elif number == calibration_answer:
            calibration.append(example)
        else:
            train.append(example)
    papers = {x.paper_id for x in examples}
    assert len({x.paragraph_id for x in test}) == len(papers)
    assert len({x.paragraph_id for x in calibration}) == len(papers)
    return train, calibration, test, calibration_answer


def model_inputs(batch: dict[str, Any], device: torch.device, labels: bool) -> dict[str, Any]:
    keys = {"input_ids", "attention_mask", "token_type_ids"}
    if labels:
        keys |= {"span_labels", "use_labels", "paragraph_masks"}
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in keys and isinstance(value, torch.Tensor)
    }


def calibration_predictions(
    model: JointDebertaTagger,
    examples: Sequence[PairExample],
    collator: JointCollator,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities, labels = [], []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(examples, batch_size=batch_size, collate_fn=collator):
            output = model(**model_inputs(batch, device, labels=False))
            probabilities.extend(output["use_logits"].softmax(-1).cpu().numpy())
            labels.extend(example.used for example in batch["examples"])
    return np.asarray(probabilities), np.asarray(labels)


def paragraph_groups(examples: Sequence[PairExample]) -> list[list[PairExample]]:
    grouped: dict[str, list[PairExample]] = {}
    for example in examples:
        grouped.setdefault(example.paragraph_id, []).append(example)
    return [grouped[key] for key in sorted(grouped)]


def add_cross_paper_negatives(
    examples: Sequence[PairExample],
    claim_sources: dict[str, tuple[str, str]],
    negatives_per_paragraph: int,
    seed: int,
) -> tuple[list[PairExample], int]:
    """Add deterministic negatives pairing each paragraph with other papers' claims."""
    if negatives_per_paragraph < 0:
        raise ValueError("negatives_per_paragraph must be non-negative")
    augmented = list(examples)
    generated = 0
    rng = random.Random(seed)
    for group in paragraph_groups(examples):
        exemplar = group[0]
        existing = {item.claim_id for item in group}
        eligible = sorted(
            claim_id for claim_id, (source_paper, _) in claim_sources.items()
            if source_paper != exemplar.paper_id and claim_id not in existing
        )
        count = min(negatives_per_paragraph, len(eligible))
        for claim_id in rng.sample(eligible, count):
            _, claim_text = claim_sources[claim_id]
            augmented.append(PairExample(
                paper_id=exemplar.paper_id,
                claim_id=claim_id,
                claim=claim_text,
                paragraph_id=exemplar.paragraph_id,
                paragraph=exemplar.paragraph,
                used=0,
                spans=tuple(),
            ))
            generated += 1
    return augmented, generated


def end_to_end_test_predictions(
    model: JointDebertaTagger,
    paragraphs: Sequence[tuple[str, str]],
    claims: dict[str, str],
    collator: JointCollator,
    conformal: SplitConformalBinary,
    segmenter: SentenceSegmenter,
    narrower: CandidateNarrower,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    """Infer claim identity, usage, and spans from paragraph text alone."""
    paragraph_predictions, violations = [], 0
    model.eval()
    for paragraph_id, paragraph in paragraphs:
        sentence_spans = segmenter(paragraph)
        retrieved, sentence_trace = narrower.retrieve_with_trace(
            [span.text for span in sentence_spans], claims
        )
        candidates = [
            PairExample("inference", claim_id, claims[claim_id],
                        paragraph_id, paragraph, 0, tuple())
            for claim_id, _ in retrieved
        ]
        candidate_records: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in DataLoader(candidates, batch_size=batch_size, collate_fn=collator):
                output = model(**model_inputs(batch, device, labels=False))
                probabilities = output["use_logits"].softmax(-1).cpu().numpy()
                span_logits = output["span_logits"].cpu().numpy()
                for row, candidate in enumerate(batch["examples"]):
                    mask = batch["paragraph_masks"][row].numpy().astype(bool)
                    record, violation = prediction_record(
                        candidate, probabilities[row], span_logits[row][mask],
                        batch["offset_mapping"][row].numpy()[mask], conformal,
                    )
                    candidate_records.append(record)
                    violations += violation
        by_claim = {item["claim_id"]: item for item in candidate_records}
        verified_ranking = [
            {
                "rank": rank,
                "claim_id": claim_id,
                "retrieval_score": score,
                "verification": by_claim[claim_id],
            }
            for rank, (claim_id, score) in enumerate(retrieved, 1)
        ]
        used = [item for item in candidate_records if item["decision"] == "used"]
        if used:
            paragraph_decision = "used"
        elif any(item["decision"] == "abstain" for item in candidate_records):
            paragraph_decision = "abstain"
        else:
            paragraph_decision = "not_used"
        paragraph_predictions.append({
            "paragraph_id": paragraph_id,
            "paragraph": paragraph,
            "decision": paragraph_decision,
            "identified_claims": [item["claim_id"] for item in used],
            "stages": {
                "sentence_segmentation": [asdict(span) for span in sentence_spans],
                "scincl_retrieval": {"top_k": narrower.top_k, "ranking": [
                    {"rank": rank, "claim_id": claim_id, "score": score}
                    for rank, (claim_id, score) in enumerate(retrieved, 1)
                ], "per_sentence": sentence_trace},
                "deberta_verification": verified_ranking,
            },
        })
    return paragraph_predictions, violations


def score_paragraph_predictions(
    paragraph_predictions: Sequence[dict[str, Any]],
    gold_examples: Sequence[PairExample],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join gold only after inference and calculate retrieval/decision metrics."""
    gold_groups = {group[0].paragraph_id: group for group in paragraph_groups(gold_examples)}
    scored_pairs: list[dict[str, Any]] = []
    retrieval_hits = gold_claims = identified_tp = identified_fp = identified_fn = 0
    correct = 0
    decision_counts = {label: 0 for label in ("used", "not_used", "abstain")}
    gold_decision_counts = {label: 0 for label in ("used", "not_used")}
    confusion = {
        gold: {predicted: 0 for predicted in ("used", "not_used", "abstain")}
        for gold in ("used", "not_used")
    }
    for paragraph_prediction in paragraph_predictions:
        paragraph_id = paragraph_prediction["paragraph_id"]
        gold_group = gold_groups[paragraph_id]
        gold_by_claim = {item.claim_id: item for item in gold_group}
        positive_gold = {item.claim_id for item in gold_group if item.used}
        retrieved = {
            item["claim_id"]
            for item in paragraph_prediction["stages"]["scincl_retrieval"]["ranking"]
        }
        identified = set(paragraph_prediction["identified_claims"])
        retrieval_hits += len(positive_gold & retrieved)
        gold_claims += len(positive_gold)
        identified_tp += len(positive_gold & identified)
        identified_fp += len(identified - positive_gold)
        identified_fn += len(positive_gold - identified)
        gold_decision = "used" if positive_gold else "not_used"
        decision = paragraph_prediction["decision"]
        decision_counts[decision] += 1
        gold_decision_counts[gold_decision] += 1
        confusion[gold_decision][decision] += 1
        correct += decision == gold_decision
        paragraph_prediction["evaluation"] = {
            "gold_decision": gold_decision,
            "gold_claim_ids": sorted(positive_gold),
            "retrieved_gold_claim_ids": sorted(positive_gold & retrieved),
            "missed_gold_claim_ids": sorted(positive_gold - retrieved),
        }
        verifications = {
            item["claim_id"]: item["verification"]
            for item in paragraph_prediction["stages"]["deberta_verification"]
        }
        for claim_id in sorted(set(gold_by_claim) | identified):
            predicted = verifications.get(claim_id, {
                "claim_id": claim_id, "paragraph_id": paragraph_id,
                "decision": "not_used", "spans": [], "p_used": 0.0,
            }).copy()
            gold = gold_by_claim.get(claim_id)
            predicted["gold_used"] = bool(gold and gold.used)
            predicted["gold_spans"] = [asdict(span) for span in gold.spans] if gold else []
            scored_pairs.append(predicted)
    total = len(paragraph_predictions)
    non_abstained = total - decision_counts["abstain"]
    metrics = {
        "paragraphs": total,
        "decision_counts": decision_counts,
        "gold_decision_counts": gold_decision_counts,
        "decision_confusion": confusion,
        "used_not_used_abstain_accuracy": correct / total,
        "coverage": non_abstained / total,
        "retrieval_recall_at_k": retrieval_hits / max(1, gold_claims),
        "claim_identification_precision": identified_tp / max(1, identified_tp + identified_fp),
        "claim_identification_recall": identified_tp / max(1, identified_tp + identified_fn),
    }
    return scored_pairs, metrics


def train_fold(
    all_examples: Sequence[PairExample],
    fold: int,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    seed: int,
    lambda_span: float,
    mu: float,
    max_length: int,
    retrieval_top_k: int,
    cross_paper_negatives: int,
    save_checkpoint: bool,
) -> dict[str, Any]:
    fold_seed = seed + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, calibration, test, calibration_answer = fold_split(all_examples, fold)
    original_train_pairs = len(train)
    original_calibration_pairs = len(calibration)
    claim_sources = {
        example.claim_id: (example.paper_id, example.claim) for example in train
    }
    train, generated_train_negatives = add_cross_paper_negatives(
        train, claim_sources, cross_paper_negatives, fold_seed + 1000
    )
    calibration, generated_calibration_negatives = add_cross_paper_negatives(
        calibration, claim_sources, cross_paper_negatives, fold_seed + 2000
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CROSS_ENCODER, use_fast=True, fix_mistral_regex=True
    )
    collator = JointCollator(tokenizer, max_length=max_length)
    model = JointDebertaTagger(lambda_span=lambda_span, mu=mu).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch, device, labels=True))
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output["loss"].detach()))
        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)
        print(f"fold={fold}/9 epoch={epoch + 1}/{epochs} loss={epoch_loss:.6f}")

    calibration_probs, calibration_labels = calibration_predictions(
        model, calibration, collator, batch_size, device
    )
    conformal = SplitConformalBinary(alpha).fit(
        calibration_probs, calibration_labels
    )
    claim_catalog = {example.claim_id: example.claim for example in train}
    scientific_encoder = ScientificEncoder(device=str(device))
    segmenter = SentenceSegmenter()
    narrower = CandidateNarrower(
        scientific_encoder,
        output_dir / f"fold_{fold}" / "scincl_claim_embeddings.pkl",
        top_k=retrieval_top_k,
    )
    test_paragraphs = [
        (group[0].paragraph_id, group[0].paragraph) for group in paragraph_groups(test)
    ]
    paragraph_predictions, violations = end_to_end_test_predictions(
        model, test_paragraphs, claim_catalog, collator, conformal, segmenter, narrower,
        batch_size, device
    )
    predictions, paragraph_metrics = score_paragraph_predictions(
        paragraph_predictions, test
    )
    scored = evaluate({"predictions": predictions})
    fold_result = {
        "pipeline_version": PIPELINE_VERSION,
        "input_contract": "paragraph_only",
        "fold": fold,
        "test_answer": fold,
        "calibration_answer": calibration_answer,
        "training_answers": [
            number for number in range(1, 10)
            if number not in {fold, calibration_answer}
        ],
        "train_pairs": len(train),
        "calibration_pairs": len(calibration),
        "test_pairs": len(test),
        "original_train_pairs": original_train_pairs,
        "original_calibration_pairs": original_calibration_pairs,
        "generated_cross_paper_train_negatives": generated_train_negatives,
        "generated_cross_paper_calibration_negatives": generated_calibration_negatives,
        "epoch_losses": epoch_losses,
        "consistency_violations": violations,
        "abstentions": paragraph_metrics["decision_counts"]["abstain"],
        "paragraph_metrics": paragraph_metrics,
        "metrics": scored["metrics"],
    }
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_predictions.json").write_text(
        json.dumps({
            "input_contract": "paragraph_only",
            "paragraph_predictions": paragraph_predictions,
            "predictions": predictions,
            "paragraph_metrics": paragraph_metrics,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (fold_dir / "character_overlap_metrics.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(fold_result, indent=2) + "\n", encoding="utf-8"
    )
    if save_checkpoint:
        torch.save(model.state_dict(), fold_dir / "joint_deberta.pt")

    del optimizer, model, scientific_encoder, narrower, segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fold_result


def aggregate(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate_metrics = {}
    for metric in METRICS_TO_AVERAGE:
        values = [float(fold["metrics"][metric]) for fold in folds]
        aggregate_metrics[metric] = {
            "mean": mean(values),
            "std": pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "completed_folds": len(folds),
        "mean_abstentions": mean(fold["abstentions"] for fold in folds),
        "total_consistency_violations": sum(
            fold["consistency_violations"] for fold in folds
        ),
        "mean_used_not_used_abstain_accuracy": mean(
            fold["paragraph_metrics"]["used_not_used_abstain_accuracy"] for fold in folds
        ),
        "metrics": aggregate_metrics,
        "folds": list(folds),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=here / "claims_and_answers_judged.json"
    )
    parser.add_argument("--output-dir", type=Path, default=here / "runs" / "9fold_cv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda-span", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument(
        "--cross-paper-negatives", type=int, default=5,
        help="Synthetic negative claims sampled from other papers per train/calibration paragraph.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--restart", action="store_true",
        help="Retrain completed folds instead of resuming from fold_summary.json files.",
    )
    args = parser.parse_args()

    examples = load_pairs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold in range(1, 10):
        saved_summary = args.output_dir / f"fold_{fold}" / "fold_summary.json"
        if saved_summary.exists() and not args.restart:
            result = json.loads(saved_summary.read_text(encoding="utf-8"))
            if result.get("pipeline_version") == PIPELINE_VERSION:
                fold_results.append(result)
                print(
                    f"Skipping completed fold {fold}/9 | "
                    f"micro_f1={result['metrics']['micro_character_f1']:.4f}"
                )
                continue
            print(f"Retraining fold {fold}/9 because its saved result predates paragraph-only inference")
        print(f"\nStarting fold {fold}/9: answer_{fold} is the test sample")
        result = train_fold(
            examples, fold, args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, alpha=args.alpha, seed=args.seed,
            lambda_span=args.lambda_span, mu=args.mu, max_length=args.max_length,
            retrieval_top_k=args.retrieval_top_k,
            cross_paper_negatives=args.cross_paper_negatives,
            save_checkpoint=args.save_checkpoints,
        )
        fold_results.append(result)
        summary = aggregate(fold_results)
        (args.output_dir / "cross_validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"fold={fold} micro_f1={result['metrics']['micro_character_f1']:.4f} "
            f"macro_f1={result['metrics']['macro_character_f1']:.4f}"
        )

    # Recompute and persist the aggregate even when every fold was resumed.
    summary = aggregate(fold_results)
    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nNine-fold averages:")
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
