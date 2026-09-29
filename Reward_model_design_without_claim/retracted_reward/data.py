from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .prompts import (
    DETECTION_PROMPT,
    MEMORY_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    MEMORY_TARGET,
    SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class Answer:
    paper_id: str
    answer_id: str
    text: str
    spans: tuple[dict, ...]
    sample_type: str = "positive"


def load_judged_data(path: str | Path) -> tuple[list[dict], list[Answer]]:
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict) or not isinstance(root.get("samples"), list):
        raise ValueError("expected a top-level samples list")
    claims, answers = [], []
    for row_number, row in enumerate(root["samples"]):
        paper_id = str(row.get("record_id", row_number))
        for number, claim in enumerate(row.get("anchor_claims", []), 1):
            claims.append({"paper_id": paper_id, "claim_number": number,
                           "text": str(claim["text"])})
        sections = (
            (("positive", row.get("positive")), ("negative", row.get("negative")))
            if "positive" in row or "negative" in row
            else (("positive", row),)
        )
        for sample_type, section in sections:
            if not isinstance(section, dict):
                continue
            for answer_id, text in section.get("answers", {}).items():
                raw_spans = section.get("judgments", {}).get(
                    answer_id, {}).get("spans", [])
                spans = []
                for span in raw_spans:
                    start, end = int(span["start"]), int(span["end"])
                    if (not (0 <= start < end <= len(text))
                            or text[start:end] != span["text"]):
                        raise ValueError(
                            f"invalid span in {paper_id}:{sample_type}:{answer_id}")
                    spans.append({"start": start, "end": end, "text": span["text"]})
                answers.append(Answer(
                    paper_id, answer_id, str(text), tuple(spans), sample_type))
    return claims, answers


def split_by_answer(answers: list[Answer], seed: int = 13) -> dict[str, list[Answer]]:
    """Reserve one answer per paper and sample type for validation and test."""
    groups: dict[tuple[str, str], list[Answer]] = {}
    for answer in answers:
        groups.setdefault((answer.paper_id, answer.sample_type), []).append(answer)
    out = {"train": [], "validation": [], "test": []}
    for (paper_id, sample_type), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: x.answer_id)
        random.Random(f"{seed}:{paper_id}:{sample_type}").shuffle(rows)
        if len(rows) < 3:
            raise ValueError(
                f"paper {paper_id} {sample_type} group needs at least three answers")
        out["test"].append(rows[0])
        out["validation"].append(rows[1])
        out["train"].extend(rows[2:])
    return out


_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9*#])")


def sentence_windows(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence windows while retaining source offsets."""
    cuts = [0] + [m.end() for m in _BOUNDARY.finditer(text)] + [len(text)]
    windows = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if start < end:
            windows.append((start, end))
    return windows


def detection_records(answers: list[Answer], negative_ratio: float | None,
                      seed: int) -> list[dict]:
    """Create local extraction examples; unannotated windows supply real negatives."""
    positive, negative = [], []
    for answer in answers:
        for start, end in sentence_windows(answer.text):
            contained = [s["text"] for s in answer.spans
                         if start <= s["start"] and s["end"] <= end]
            # If an annotation crosses a sentence boundary, expand to the full answer
            # rather than silently creating a false negative.
            crossing = any(s["start"] < end and s["end"] > start and
                           not (start <= s["start"] and s["end"] <= end)
                           for s in answer.spans)
            if crossing:
                continue
            record = conversation_record(
                SYSTEM_PROMPT, DETECTION_PROMPT.format(text=answer.text[start:end]),
                json.dumps({"uses_retracted_claim": bool(contained), "spans": contained},
                           ensure_ascii=False), "detection")
            (positive if contained else negative).append(record)
    if negative_ratio is not None:
        rng = random.Random(seed)
        rng.shuffle(negative)
        negative = negative[:round(len(positive) * negative_ratio)]
    result = positive + negative
    random.Random(seed).shuffle(result)
    return result


def conversation_record(system: str, user: str, assistant: str, task: str) -> dict:
    return {"task": task, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def memory_records(claims: list[dict]) -> list[dict]:
    return [conversation_record(MEMORY_SYSTEM_PROMPT,
            MEMORY_PROMPT.format(claim=c["text"]), MEMORY_TARGET, "memory")
            for c in claims]


def paragraph_record(answer: Answer) -> dict:
    """One closed-book detection example containing the complete paragraph."""
    span_texts = [span["text"] for span in answer.spans]
    return conversation_record(
        SYSTEM_PROMPT, DETECTION_PROMPT.format(text=answer.text),
        json.dumps({"uses_retracted_claim": bool(span_texts), "spans": span_texts},
                   ensure_ascii=False),
        "negative_detection" if answer.sample_type == "negative" else "detection")


def negative_sentence_records(answers: list[Answer], maximum: int,
                              seed: int) -> list[dict]:
    """Select hard negatives from unannotated sentences of the same papers."""
    candidates = []
    for answer in answers:
        for start, end in sentence_windows(answer.text):
            overlaps = any(span["start"] < end and span["end"] > start
                           for span in answer.spans)
            if not overlaps:
                candidates.append(conversation_record(
                    SYSTEM_PROMPT,
                    DETECTION_PROMPT.format(text=answer.text[start:end]),
                    json.dumps({"uses_retracted_claim": False, "spans": []}),
                    "negative_detection"))
    random.Random(seed).shuffle(candidates)
    return candidates[:maximum]


def replay(records: list[dict], memories: list[dict], memory_ratio: float,
           seed: int) -> list[dict]:
    """Interleave a target memory:detection ratio using deterministic replay."""
    if memory_ratio < 0:
        raise ValueError("memory_ratio must be non-negative")
    rng = random.Random(seed)
    count = round(len(records) * memory_ratio)
    sampled = [memories[i % len(memories)] for i in range(count)] if memories else []
    rng.shuffle(sampled)
    mixed, stride = [], memory_ratio
    memory_cursor = 0
    accumulator = 0.0
    for record in records:
        mixed.append(record)
        accumulator += stride
        while accumulator >= 1 and memory_cursor < len(sampled):
            mixed.append(sampled[memory_cursor]); memory_cursor += 1; accumulator -= 1
    mixed.extend(sampled[memory_cursor:])
    return mixed


def build_splits(path: str | Path, seed: int = 13, negative_ratio: float = 0.5,
                 memory_ratio: float | None = None) -> dict[str, list[dict]]:
    """Build paper-local curriculum blocks with positive and negative answers.

    ``memory_ratio`` is accepted only for compatibility with older commands;
    the core ratio is now determined by the data (normally 5:7 per paper).
    """
    claims, answers = load_judged_data(path)
    split = split_by_answer(answers, seed)
    claims_by_paper: dict[str, list[dict]] = {}
    for claim in claims:
        claims_by_paper.setdefault(claim["paper_id"], []).append(claim)
    train_by_paper: dict[str, list[Answer]] = {}
    for answer in split["train"]:
        train_by_paper.setdefault(answer.paper_id, []).append(answer)
    train = []
    for paper_id in sorted(train_by_paper):
        paper_answers = sorted(train_by_paper[paper_id], key=lambda x: x.answer_id)
        # Ordering is intentional: first encode this paper's claim registry,
        # then apply it to its held-in positive and negative paragraphs.
        paper_memories = memory_records(claims_by_paper.get(paper_id, []))
        paper_detections = [paragraph_record(answer) for answer in paper_answers]
        negative_count = round(len(paper_answers) * negative_ratio)
        paper_negatives = negative_sentence_records(
            paper_answers, negative_count,
            seed + int(paper_id) if paper_id.isdigit() else seed)
        for record in paper_memories + paper_detections + paper_negatives:
            train.append({**record, "paper_id": paper_id})

    def tagged_paragraph(answer: Answer) -> dict:
        return {**paragraph_record(answer), "paper_id": answer.paper_id}
    return {
        "train": train,
        "validation": [tagged_paragraph(answer) for answer in split["validation"]],
        "test": [tagged_paragraph(answer) for answer in split["test"]],
    }
