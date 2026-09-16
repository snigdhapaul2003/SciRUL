"""Run the optimized evaluation pipeline with an unmodified base Llama model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_inference import main as run_inference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base-model-only inference; LoRA/adapters cannot be supplied."
    )
    parser.add_argument("input", help="Paragraph-only inference JSON")
    parser.add_argument("checkpoint", help="Base Llama checkpoint path or Hugging Face ID")
    parser.add_argument("output", help="Prediction JSON destination")
    parser.add_argument("--index", required=True, help="Frozen claim-index NPZ")
    parser.add_argument("--conformal")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--semantic-threshold", type=float, default=.55)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    forwarded = [
        args.input,
        args.checkpoint,
        args.output,
        "--index", args.index,
        "--top-k", str(args.top_k),
        "--samples", str(args.samples),
        "--semantic-threshold", str(args.semantic_threshold),
        "--confidence-threshold", str(args.confidence_threshold),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.conformal:
        forwarded.extend(["--conformal", args.conformal])
    # Deliberately do not forward --adapter: only frozen base weights are loaded.
    run_inference(forwarded)


if __name__ == "__main__":
    main()
