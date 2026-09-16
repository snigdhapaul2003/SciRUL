from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    claim_numbers: tuple[int, ...]

    def validate(self, paragraph: str) -> None:
        assert 0 <= self.start < self.end <= len(paragraph)
        assert paragraph[self.start:self.end] == self.text, (self.start, self.end)


@dataclass(frozen=True)
class Claim:
    paper_id: str
    claim_number: int
    text: str

    @property
    def claim_id(self) -> str:
        return f"{self.paper_id}:claim_{self.claim_number}"


@dataclass(frozen=True)
class Example:
    paper_id: str
    paragraph_id: str
    paragraph: str
    claims: tuple[Claim, ...]
    spans: tuple[Span, ...]
    answer_model: str


def load_dataset(path: str | Path) -> list[Example]:
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict) or not isinstance(root.get("samples"), list):
        raise ValueError("input must contain a top-level 'samples' list")
    out: list[Example] = []
    for row_index, row in enumerate(root["samples"]):
        paper_id = str(row.get("record_id", row_index))
        claims = tuple(
            Claim(paper_id, claim_number, str(claim["text"]))
            for claim_number, claim in enumerate(row["anchor_claims"], 1)
        )
        if len(claims) != 5:
            raise ValueError(f"{paper_id}: expected exactly five sibling claims")
        for answer_key, paragraph in row["answers"].items():
            judgment = row.get("judgments", {}).get(answer_key, {})
            spans = tuple(
                Span(
                    int(raw_span["start"]),
                    int(raw_span["end"]),
                    str(raw_span["text"]),
                    tuple(map(int, raw_span["claim_numbers"])),
                )
                for raw_span in judgment.get("spans", [])
            )
            for span in spans:
                span.validate(paragraph)
            out.append(
                Example(
                    paper_id=paper_id,
                    paragraph_id=f"{paper_id}:{answer_key}",
                    paragraph=paragraph,
                    claims=claims,
                    spans=spans,
                    answer_model=str(
                        row.get("answer_models", {}).get(answer_key, "unknown")
                    ),
                )
            )
    return out


def split_answers(examples: Iterable[Example], seed: int = 13) -> dict[str, list[Example]]:
    """Allocate seven train, one calibration, one test answer for every paper."""
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.paper_id, []).append(ex)
    result = {"train": [], "calibration": [], "test": []}
    for paper_id, rows in sorted(groups.items()):
        if len(rows) != 9:
            raise ValueError(f"{paper_id}: expected 9 answers, got {len(rows)}")
        rows = sorted(rows, key=lambda x: x.paragraph_id)
        random.Random(f"{seed}:{paper_id}").shuffle(rows)
        result["test"].append(rows[0])
        result["calibration"].append(rows[1])
        result["train"].extend(rows[2:])
    ids = [{x.paragraph_id for x in result[k]} for k in result]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    return result


_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9*#])")
_ABBREVIATIONS = {"e.g.", "i.e.", "et al.", "Fig.", "Dr.", "vs.", "cf."}


def sentences_with_offsets(paragraph: str) -> list[tuple[int, int, str]]:
    """Conservative sentence splitter preserving authoritative paragraph offsets."""
    cuts = [0]
    for match in _BOUNDARY.finditer(paragraph):
        left = paragraph[max(0, match.start() - 12):match.start()].strip()
        if any(left.endswith(a) for a in _ABBREVIATIONS):
            continue
        cuts.append(match.end())
    cuts.append(len(paragraph))
    sentences = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and paragraph[start].isspace():
            start += 1
        while end > start and paragraph[end - 1].isspace():
            end -= 1
        if start < end:
            sentences.append((start, end, paragraph[start:end]))
    return sentences


def to_jsonable(ex: Example) -> dict[str, Any]:
    return asdict(ex)
