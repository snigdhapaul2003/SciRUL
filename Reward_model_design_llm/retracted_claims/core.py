from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass
class Step:
    span: str
    # Self-consistency vote fraction used by conformal acceptance.
    score: float = 0.0
    semantic_similarity: float = 0.0
    status: str = "unverified"
    start: int | None = None
    end: int | None = None


def parse_derivation(text: str, sentence: str) -> list[Step]:
    parsed_json = json.loads(text)
    if not isinstance(parsed_json, dict) or set(parsed_json) != {"steps"}:
        raise ValueError("top-level JSON must contain only 'steps'")
    raw_steps = parsed_json.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list")

    steps = []
    for item in raw_steps:
        if not isinstance(item, dict) or not isinstance(item.get("span"), str):
            raise ValueError("invalid step")
        if set(item) != {"span"}:
            raise ValueError("each step must contain only 'span'")
        span = item["span"]
        if not span:
            raise ValueError("steps must use non-empty spans; use an empty list for none")
        if sentence.find(span) < 0:
            raise ValueError("span is not verbatim")
        steps.append(Step(span))
    return steps


class Scorer(Protocol):
    def score(self, claim: str, span: str) -> float: ...


class CosineVerifier:
    def __init__(self, model_name="malteos/scincl", threshold=0.55, encoder=None):
        self.model_name = model_name
        self.threshold = threshold
        self.encoder = encoder

    def score(self, claim: str, span: str) -> float:
        from .retrieval import ScientificIndex

        encoder = self.encoder or ScientificIndex(self.model_name)
        embeddings = encoder._encode([claim, span])
        return float(embeddings[0] @ embeddings[1])

    def verify(
        self,
        claim: str,
        sentence: str,
        steps: list[Step],
        paragraph_start: int = 0,
    ) -> list[Step]:
        for step in steps:
            if not step.span:
                step.status = "verified_negative"
                continue

            local_start = sentence.find(step.span)
            if local_start < 0:
                step.status = "invalid"
                continue

            step.start = paragraph_start + local_start
            step.end = step.start + len(step.span)
            step.semantic_similarity = self.score(claim, step.span)
            step.status = (
                "verified_positive"
                if step.semantic_similarity >= self.threshold
                else "unverified"
            )
        return steps

    def verify_batch(self, items: list[tuple[str, str, list[Step], int]]):
        pending = []
        for claim, sentence, steps, paragraph_start in items:
            for step in steps:
                if not step.span:
                    step.status = "verified_negative"
                    continue

                local_start = sentence.find(step.span)
                if local_start < 0:
                    step.status = "invalid"
                    continue

                step.start = paragraph_start + local_start
                step.end = step.start + len(step.span)
                pending.append((claim, step))

        from .retrieval import ScientificIndex

        encoder = self.encoder or ScientificIndex(self.model_name)
        scores = encoder.similarities([(claim, step.span) for claim, step in pending])
        for (_, step), score in zip(pending, scores):
            step.semantic_similarity = score
            step.status = (
                "verified_positive" if score >= self.threshold else "unverified"
            )
        return items


def conformal_threshold(calibration_scores: list[float], alpha=0.05) -> float:
    if not calibration_scores:
        raise ValueError("empty calibration scores")
    sample_count = len(calibration_scores)
    rank = min(
        sample_count,
        math.ceil((sample_count + 1) * (1 - alpha)),
    )
    return float(np.sort(np.asarray(calibration_scores))[rank - 1])


def aggregate(steps: list[Step], threshold: float) -> str:
    if not steps:
        return "not_used"
    valid = [s for s in steps if s.status != "invalid"]
    if any(
        step.status == "verified_positive" and step.score >= threshold
        for step in valid
    ):
        return "used"
    if valid and all(step.status == "verified_negative" for step in valid):
        return "not_used"
    return "abstain"


def step_dict(step: Step, paragraph: str):
    if step.start is not None:
        assert paragraph[step.start:step.end] == step.span
    return asdict(step)
