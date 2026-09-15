"""Deterministic retracted-claim usage and localization research prototype.

No LLM is used.  Fixed model weights and inputs produce deterministic inference.
Character offsets are authoritative throughout storage and output; BIO labels exist
only transiently inside ``JointCollator``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# SciNCL exposes ordinary Hugging Face embeddings suitable for sentence-level
# mean-pooled retrieval.
SCIENTIFIC_ENCODER = "malteos/scincl"
CROSS_ENCODER = "microsoft/deberta-v3-base"
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    text: str

    def validate(self, paragraph: str) -> None:
        assert 0 <= self.start < self.end <= len(paragraph)
        assert paragraph[self.start : self.end] == self.text


@dataclass(frozen=True)
class PairExample:
    paper_id: str
    claim_id: str
    claim: str
    paragraph_id: str
    paragraph: str
    used: int
    spans: tuple[CharSpan, ...]


def load_pairs(path: Path) -> list[PairExample]:
    """Adapt the inspected schema and validate every gold character span."""
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict) or not isinstance(root.get("samples"), list):
        raise ValueError("Expected the inspected top-level samples list")
    pairs: list[PairExample] = []
    for sample_index, sample in enumerate(root["samples"]):
        if not isinstance(sample, dict):
            continue
        paper_id = str(sample.get("record_id", sample_index))
        raw_claims = sample.get("anchor_claims", [])
        claims = [c.get("text") if isinstance(c, dict) else c for c in raw_claims]
        answers, judgments = sample.get("answers", {}), sample.get("judgments", {})
        if not isinstance(answers, dict) or not isinstance(judgments, dict):
            continue
        for answer_key, paragraph in answers.items():
            judgment = judgments.get(answer_key, {})
            if not isinstance(paragraph, str) or not isinstance(judgment, dict):
                continue
            raw_spans = judgment.get("spans", [])
            for claim_index, claim in enumerate(claims, 1):
                if not isinstance(claim, str) or not claim.strip():
                    continue
                spans: list[CharSpan] = []
                for raw in raw_spans if isinstance(raw_spans, list) else []:
                    if not isinstance(raw, dict) or claim_index not in raw.get("claim_numbers", []):
                        continue
                    span = CharSpan(int(raw["start"]), int(raw["end"]), str(raw["text"]))
                    span.validate(paragraph)
                    spans.append(span)
                pairs.append(
                    PairExample(
                        paper_id=paper_id,
                        claim_id=f"{paper_id}:claim_{claim_index}",
                        claim=claim,
                        paragraph_id=f"{paper_id}:{answer_key}",
                        paragraph=paragraph,
                        used=int(bool(spans)),
                        spans=tuple(spans),
                    )
                )
    if not pairs:
        raise ValueError("No claim-paragraph pairs could be adapted")
    return pairs


def grouped_split(
    examples: Sequence[PairExample], seed: int = 13, train=0.7, calibration=0.15
) -> tuple[list[PairExample], list[PairExample], list[PairExample]]:
    """Split by paper, preventing claim/paragraph leakage across partitions."""
    paper_ids = sorted({x.paper_id for x in examples})
    random.Random(seed).shuffle(paper_ids)
    n = len(paper_ids)
    a, b = max(1, round(n * train)), max(2, round(n * (train + calibration)))
    groups = set(paper_ids[:a]), set(paper_ids[a:b]), set(paper_ids[b:])
    return tuple([[x for x in examples if x.paper_id in group] for group in groups])  # type: ignore[return-value]


def one_paragraph_per_record_split(
    examples: Sequence[PairExample], seed: int = 13
) -> tuple[list[PairExample], list[PairExample]]:
    """Hold out exactly one whole paragraph from every paper record.

    The selection is seeded and all claim pairs for an answer remain in the same
    partition. Every other paragraph from that record is assigned to training.
    """
    by_paper: dict[str, dict[str, list[PairExample]]] = {}
    for example in examples:
        by_paper.setdefault(example.paper_id, {}).setdefault(
            example.paragraph_id, []
        ).append(example)

    rng = random.Random(seed)
    train_examples: list[PairExample] = []
    test_examples: list[PairExample] = []
    for paper_id in sorted(by_paper):
        paragraphs = sorted(by_paper[paper_id])
        if len(paragraphs) < 2:
            raise ValueError(
                f"Paper {paper_id!r} needs at least two paragraphs for train/test"
            )
        test_paragraph = rng.choice(paragraphs)
        for paragraph_id, paragraph_examples in by_paper[paper_id].items():
            destination = (
                test_examples if paragraph_id == test_paragraph else train_examples
            )
            destination.extend(paragraph_examples)

    train_ids = {x.paragraph_id for x in train_examples}
    test_ids = {x.paragraph_id for x in test_examples}
    assert train_ids.isdisjoint(test_ids)
    assert len(test_ids) == len(by_paper)
    return train_examples, test_examples


def write_pair_split(path: Path, examples: Sequence[PairExample]) -> None:
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        **asdict(example),
                        "spans": [asdict(span) for span in example.spans],
                    }
                    for example in examples
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_splits(input_path: Path, output_dir: Path, seed: int) -> None:
    pairs = load_pairs(input_path)
    train_examples, test_examples = one_paragraph_per_record_split(pairs, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_pair_split(output_dir / "train.json", train_examples)
    write_pair_split(output_dir / "test.json", test_examples)
    train_paragraphs = {x.paragraph_id for x in train_examples}
    test_paragraphs = {x.paragraph_id for x in test_examples}
    statistics = {
        "seed": seed,
        "records": len({x.paper_id for x in pairs}),
        "train_paragraphs": len(train_paragraphs),
        "test_paragraphs": len(test_paragraphs),
        "train_claim_paragraph_pairs": len(train_examples),
        "test_claim_paragraph_pairs": len(test_examples),
        "test_samples_per_record": 1,
    }
    (output_dir / "split_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2))


class SentenceSegmenter:
    """Split paragraphs into sentence-like units while preserving exact offsets."""

    def __call__(self, paragraph: str) -> list[CharSpan]:
        spans: list[CharSpan] = []
        # Avoid splitting common scholarly abbreviations while allowing a
        # lowercase sentence starter. Offsets always refer to the source text.
        abbreviations = {
            "e.g.", "i.e.", "et al.", "fig.", "eq.", "ref.", "dr.", "mr.",
            "mrs.", "ms.", "prof.", "vs.", "no.", "inc.", "approx.",
        }
        boundaries = [0]
        for match in re.finditer(r"[.!?][\"')\]]*(?=\s+|$)", paragraph):
            prefix = paragraph[: match.end()].lower().rstrip("\"')]")
            if any(prefix.endswith(item) for item in abbreviations):
                continue
            boundaries.append(match.end())
        boundaries.append(len(paragraph))
        for left, right in zip(boundaries, boundaries[1:]):
            raw = paragraph[left:right]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start, end = left + leading, left + trailing
            if end <= start:
                continue
            span = CharSpan(start, end, paragraph[start:end])
            span.validate(paragraph)
            spans.append(span)
        return spans or [CharSpan(0, len(paragraph), paragraph)]


class ScientificEncoder(nn.Module):
    def __init__(self, model_name: str = SCIENTIFIC_ENCODER, device: str = "cpu") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = torch.device(device)
        # SciNCL's tokenizer advertises an effectively unlimited length even
        # though its BERT encoder has only 512 positional embeddings.
        self.max_length = int(self.model.config.max_position_embeddings)

    @torch.inference_mode()
    def token_vectors(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = batch.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**encoded).last_hidden_state
        special = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
             for ids in encoded["input_ids"].cpu().tolist()], device=self.device
        )
        mask = encoded["attention_mask"].bool() & ~special.bool()
        return F.normalize(hidden, dim=-1), mask, offsets

    @torch.inference_mode()
    def pooled(self, texts: Sequence[str]) -> np.ndarray:
        vectors, mask, _ = self.token_vectors(texts)
        pooled = (vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(pooled, dim=-1).cpu().numpy()


class CandidateNarrower:
    def __init__(self, encoder: ScientificEncoder, cache_path: Path, top_k: int = 10) -> None:
        self.encoder, self.cache_path, self.top_k = encoder, cache_path, top_k

    def claim_embeddings(self, claims: dict[str, str]) -> tuple[list[str], np.ndarray]:
        digest = hashlib.sha256(json.dumps(claims, sort_keys=True).encode()).hexdigest()
        if self.cache_path.exists():
            cached = pickle.loads(self.cache_path.read_bytes())
            if cached.get("digest") == digest:
                return cached["ids"], cached["embeddings"]
        ids = list(claims)
        embeddings = self.encoder.pooled([claims[i] for i in ids])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(pickle.dumps({"digest": digest, "ids": ids, "embeddings": embeddings}))
        return ids, embeddings

    def retrieve_with_trace(
        self, sentences: Sequence[str], claims: dict[str, str]
    ) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
        ids, claim_vectors = self.claim_embeddings(claims)
        sentence_vectors = self.encoder.pooled(sentences)
        similarity = sentence_vectors @ claim_vectors.T
        scores = similarity.max(axis=0)
        order = np.argsort(-scores)[: self.top_k]
        ranked = [(ids[i], float(scores[i])) for i in order]
        trace = [
            {
                "sentence_index": sentence_index,
                "text": sentence,
                "top_claims": [
                    {"claim_id": ids[i], "score": float(similarity[sentence_index, i])}
                    for i in np.argsort(-similarity[sentence_index])[: self.top_k]
                ],
            }
            for sentence_index, sentence in enumerate(sentences)
        ]
        return ranked, trace

class JointCollator:
    """Convert gold character offsets to paragraph-token BIO labels only here."""

    def __init__(self, tokenizer: Any, max_length: int = 512) -> None:
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, examples: Sequence[PairExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [x.claim for x in examples], [x.paragraph for x in examples],
            padding=True, truncation="only_second", max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        labels = torch.full_like(encoded["input_ids"], -100)
        paragraph_masks = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        for row, example in enumerate(examples):
            previous_span: CharSpan | None = None
            for token_index, ((start, end), sequence_id) in enumerate(zip(offsets[row].tolist(), encoded.sequence_ids(row))):
                if sequence_id != 1 or end <= start:
                    continue
                paragraph_masks[row, token_index] = True
                overlapping = next((s for s in example.spans if start < s.end and end > s.start), None)
                if overlapping is None:
                    labels[row, token_index] = LABEL_O
                    previous_span = None
                else:
                    labels[row, token_index] = LABEL_I if previous_span == overlapping else LABEL_B
                    previous_span = overlapping
        return {
            **encoded,
            "span_labels": labels,
            "use_labels": torch.tensor([x.used for x in examples], dtype=torch.long),
            "paragraph_masks": paragraph_masks,
            "offset_mapping": offsets,
            "examples": examples,
        }


class JointDebertaTagger(nn.Module):
    def __init__(self, model_name: str = CROSS_ENCODER, lambda_span: float = 1.0, mu: float = 0.5) -> None:
        super().__init__()
        # Train from an FP32 master copy. Some recent Transformers releases
        # preserve this checkpoint's FP16 storage dtype; naive FP16 optimization
        # without gradient scaling is unstable even when its forward pass works.
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden = self.encoder.config.hidden_size
        # Match custom heads to the encoder dtype explicitly so future checkpoint
        # dtype changes cannot cause Half-vs-Float matrix multiplication failures.
        encoder_dtype = next(self.encoder.parameters()).dtype
        self.span_head = nn.Linear(hidden, 3, dtype=encoder_dtype)
        self.usage_head = nn.Linear(hidden, 2, dtype=encoder_dtype)
        self.lambda_span, self.mu = lambda_span, mu

    def forward(self, input_ids, attention_mask, token_type_ids=None, span_labels=None,
                use_labels=None, paragraph_masks=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        span_logits, use_logits = self.span_head(hidden), self.usage_head(hidden[:, 0])
        loss = None
        if span_labels is not None and use_labels is not None:
            use_loss = F.cross_entropy(use_logits, use_labels)
            span_loss = F.cross_entropy(span_logits.view(-1, 3), span_labels.view(-1), ignore_index=-100)
            p_use = use_logits.softmax(-1)[:, 1]
            p_non_o = 1 - span_logits.softmax(-1)[..., LABEL_O]
            p_non_o = p_non_o.masked_fill(~paragraph_masks, 0).max(dim=1).values
            consistency = F.relu(p_use - p_non_o).mean()
            loss = use_loss + self.lambda_span * span_loss + self.mu * consistency
        return {"loss": loss, "use_logits": use_logits, "span_logits": span_logits}


def constrained_viterbi(logits: np.ndarray) -> list[int]:
    """Decode BIO while forbidding O->I and sequence-initial I."""
    transitions = np.zeros((3, 3), dtype=np.float32)
    transitions[LABEL_O, LABEL_I] = -np.inf
    score = logits[0].copy()
    score[LABEL_I] = -np.inf
    back = np.zeros((len(logits), 3), dtype=np.int64)
    for t in range(1, len(logits)):
        candidates = score[:, None] + transitions
        back[t] = candidates.argmax(0)
        score = candidates.max(0) + logits[t]
    path = [int(score.argmax())]
    for t in range(len(logits) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def decode_spans(paragraph: str, offsets: Sequence[Sequence[int]], labels: Sequence[int]) -> list[CharSpan]:
    runs: list[list[int]] = []
    for (start, end), label in zip(offsets, labels):
        # Tokenizer offsets may be NumPy int64 values, which json.dumps cannot
        # serialize. Character offsets in all public records are native ints.
        start, end = int(start), int(end)
        if label == LABEL_O or end <= start:
            continue
        if label == LABEL_B or not runs:
            runs.append([start, end])
        else:
            runs[-1][1] = end
    merged: list[list[int]] = []
    for run in runs:
        if merged:
            gap = paragraph[merged[-1][1] : run[0]]
            if not gap.strip() or (len(gap.strip()) == 1 and gap.strip() in string.punctuation):
                merged[-1][1] = run[1]
                continue
        merged.append(run)
    result = [CharSpan(s, e, paragraph[s:e]) for s, e in merged]
    for span in result:
        span.validate(paragraph)
    return result


class SplitConformalBinary:
    """Class-conditional split conformal predictor; ambiguous sets abstain."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.quantiles: dict[int, float] = {}

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SplitConformalBinary":
        for cls in (0, 1):
            scores = 1 - probabilities[labels == cls, cls]
            if not len(scores):
                raise ValueError(f"Calibration contains no class {cls} examples")
            level = min(1.0, math.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
            self.quantiles[cls] = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, probabilities: Sequence[float]) -> tuple[str, list[int]]:
        prediction_set = [c for c in (0, 1) if 1 - probabilities[c] <= self.quantiles[c]]
        return (("not_used" if prediction_set == [0] else "used") if len(prediction_set) == 1 else "abstain", prediction_set)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"alpha": self.alpha, "quantiles": self.quantiles}, indent=2), encoding="utf-8")


def prediction_record(
    example: PairExample, probabilities: Sequence[float], span_logits: np.ndarray,
    paragraph_offsets: Sequence[Sequence[int]], conformal: SplitConformalBinary,
) -> tuple[dict[str, Any], int]:
    decision, prediction_set = conformal.predict(probabilities)
    labels = constrained_viterbi(span_logits)
    spans = decode_spans(example.paragraph, paragraph_offsets, labels)
    violation = int(decision == "used" and not spans)
    if violation:
        decision = "abstain"
    output = {
        "claim_id": example.claim_id,
        "paragraph_id": example.paragraph_id,
        "decision": decision,
        "conformal_set": prediction_set,
        "p_used": float(probabilities[1]),
        "spans": [asdict(s) for s in spans] if decision == "used" else [],
    }
    for span in output["spans"]:
        assert example.paragraph[span["start"] : span["end"]] == span["text"]
    return output, violation


def inspect_adapter(input_path: Path) -> None:
    pairs = load_pairs(input_path)
    print(json.dumps({
        "pairs": len(pairs), "papers": len({x.paper_id for x in pairs}),
        "claims": len({x.claim_id for x in pairs}), "paragraphs": len({x.paragraph_id for x in pairs}),
        "positive_pairs": sum(x.used for x in pairs), "negative_pairs": sum(not x.used for x in pairs),
        "gold_spans": sum(len(x.spans) for x in pairs),
    }, indent=2))


def train_joint(
    input_path: Path, output_dir: Path, *, epochs: int, batch_size: int,
    learning_rate: float, alpha: float, seed: int,
) -> None:
    """Train Stage 3, calibrate on held-out papers, and emit test predictions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(input_path)
    training_pool, test_pairs = one_paragraph_per_record_split(pairs, seed=seed)
    # Split conformal requires data not used for gradient updates. The externally
    # visible split still has eight training-pool paragraphs and one test paragraph
    # per record; one training-pool paragraph per record is used only for calibration.
    train_pairs, calibration_pairs = one_paragraph_per_record_split(
        training_pool, seed=seed + 1
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CROSS_ENCODER, use_fast=True, fix_mistral_regex=True
    )
    collator = JointCollator(tokenizer)
    model = JointDebertaTagger().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_path = output_dir / "joint_deberta.pt"
    if epochs == 0:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot use --epochs 0 without an existing checkpoint: {checkpoint_path}"
            )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"Loaded existing checkpoint from {checkpoint_path}")

    loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, collate_fn=collator)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = {
                key: value.to(device) for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids", "span_labels", "use_labels", "paragraph_masks"}
            }
            output = model(**inputs)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output["loss"].detach()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}")

    def probabilities_for(items: Sequence[PairExample]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probabilities, labels = [], []
        with torch.inference_mode():
            for batch in DataLoader(items, batch_size=batch_size, collate_fn=collator):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch.get("token_type_ids", None).to(device) if batch.get("token_type_ids") is not None else None,
                )
                probabilities.extend(output["use_logits"].softmax(-1).cpu().numpy())
                labels.extend(x.used for x in batch["examples"])
        return np.asarray(probabilities), np.asarray(labels)

    calibration_probabilities, calibration_labels = probabilities_for(calibration_pairs)
    conformal = SplitConformalBinary(alpha).fit(calibration_probabilities, calibration_labels)
    predictions, violations = [], 0
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(test_pairs, batch_size=batch_size, collate_fn=collator):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids", None).to(device) if batch.get("token_type_ids") is not None else None,
            )
            use_probabilities = output["use_logits"].softmax(-1).cpu().numpy()
            span_logits = output["span_logits"].cpu().numpy()
            for row, example in enumerate(batch["examples"]):
                mask = batch["paragraph_masks"][row].numpy().astype(bool)
                record, violation = prediction_record(
                    example, use_probabilities[row], span_logits[row][mask],
                    batch["offset_mapping"][row].numpy()[mask], conformal,
                )
                record["gold_used"] = bool(example.used)
                record["gold_spans"] = [asdict(s) for s in example.spans]
                predictions.append(record)
                violations += violation

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    conformal.save(output_dir / "conformal.json")
    (output_dir / "test_predictions.json").write_text(json.dumps({
        "predictions": predictions,
        "metrics": {
            "consistency_violations": violations,
            "test_pairs": len(test_pairs),
            "abstentions": sum(x["decision"] == "abstain" for x in predictions),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint, calibration, and predictions to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect-adapter")
    inspect_parser.add_argument("input", type=Path)
    train_parser = sub.add_parser("train-stage3")
    train_parser.add_argument("input", type=Path)
    train_parser.add_argument("output_dir", type=Path)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--alpha", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=13)
    split_parser = sub.add_parser("build-splits")
    split_parser.add_argument("input", type=Path)
    split_parser.add_argument("output_dir", type=Path)
    split_parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.command == "inspect-adapter":
        inspect_adapter(args.input)
    elif args.command == "train-stage3":
        train_joint(
            args.input, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, alpha=args.alpha, seed=args.seed,
        )
    elif args.command == "build-splits":
        build_splits(args.input, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
