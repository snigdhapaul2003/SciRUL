"""Build and cache the frozen scientific claim-embedding index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retracted_claims.data import load_dataset
from retracted_claims.retrieval import ScientificIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("cache_dir")
    parser.add_argument("--encoder", default="malteos/scincl")
    args = parser.parse_args()

    examples = load_dataset(args.input)
    claims_by_id = {
        claim.claim_id: claim
        for example in examples
        for claim in example.claims
    }

    index = ScientificIndex(args.encoder)
    cache_path = index.build(list(claims_by_id.values()), args.cache_dir)
    print(cache_path)


if __name__ == "__main__":
    main()
