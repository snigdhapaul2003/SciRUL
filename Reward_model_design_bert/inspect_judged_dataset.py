"""Inspect an unknown JSON dataset without assuming a fixed schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def walk(value: Any, path: str, types: dict[str, Counter[str]], examples: dict[str, list[Any]]) -> None:
    kind = type(value).__name__
    types[path][kind] += 1
    if len(examples[path]) < 3 and not isinstance(value, (dict, list)):
        examples[path].append(value if not isinstance(value, str) else value[:160])
    if isinstance(value, dict):
        for key, child in value.items():
            walk(child, f"{path}.{key}", types, examples)
    elif isinstance(value, list):
        for child in value:
            walk(child, f"{path}[]", types, examples)


def inspect(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    types: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = defaultdict(list)
    walk(document, "$", types, examples)
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "root_type": type(document).__name__,
        "paths": {
            key: {"types": dict(counter), "examples": examples.get(key, [])}
            for key, counter in sorted(types.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively inspect a JSON dataset schema.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Inspection report written to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
