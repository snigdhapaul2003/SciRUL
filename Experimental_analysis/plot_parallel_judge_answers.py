"""Plot parallel QWA judge metrics by prompt type and source model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plot_judge_answers as plotting


# Reuse the established plotting and CSV layout with the parallel judge schema.
plotting.METRICS = {
    "uses_retracted_claim": {
        "title": "Uses retracted-paper claim",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#C94C4C", "no": "#278C82"},
        "filename": "uses_retracted_claim",
    },
    "uses_valid_claim": {
        "title": "Uses valid-paper claim",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#3975B7", "no": "#8A9099"},
        "filename": "uses_valid_claim",
    },
    "uses_both": {
        "title": "Uses both retracted and valid claims",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#8C5AA8", "no": "#8A9099"},
        "filename": "uses_both",
    },
    "uses_none": {
        "title": "Uses neither claim set",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#E3A72F", "no": "#8A9099"},
        "filename": "uses_none",
    },
    "mentions_retracted_paper_as_retracted": {
        "title": "Explicitly identifies paper as retracted",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#3975B7", "no": "#8A9099"},
        "filename": "explicit_retraction_mention",
    },
}


def extract_results(
    document: dict[str, Any],
) -> tuple[list[str], list[int], dict[tuple[int, str, str], Counter[str]]]:
    """Collect metric counts indexed by prompt type and source model."""
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError("Input JSON must contain a 'results' list")

    models: list[str] = []
    prompt_types: set[int] = set()
    counts: dict[tuple[int, str, str], Counter[str]] = defaultdict(Counter)

    for result in results:
        if not isinstance(result, dict):
            continue
        model_outputs = result.get("model_outputs", [])
        if not isinstance(model_outputs, list):
            continue
        for model_output in model_outputs:
            if not isinstance(model_output, dict):
                continue
            model = model_output.get("model_name")
            if not isinstance(model, str) or not model:
                continue
            if model not in models:
                models.append(model)

            for key, answer in model_output.items():
                if not key.startswith("judge_answer_") or not isinstance(answer, dict):
                    continue
                suffix = key.removeprefix("judge_answer_")
                if not suffix.isdigit():
                    continue
                prompt_type = int(suffix)
                prompt_types.add(prompt_type)
                for metric, settings in plotting.METRICS.items():
                    status = answer.get(metric)
                    if status not in settings["statuses"]:
                        raise ValueError(
                            f"Invalid {metric} value {status!r} for {model}, "
                            f"prompt type {prompt_type}"
                        )
                    counts[(prompt_type, model, metric)][status] += 1

    if not models or not prompt_types:
        raise ValueError("No valid judge_answer_n objects were found")
    return models, sorted(prompt_types), counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot parallel QWA judge metrics by prompt type and source model."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("parallel_QWA_judge_answers.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plot_parallel")
    )
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    models, prompt_types, counts = extract_results(document)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = plotting.save_individual_plots(
        args.output_dir, models, prompt_types, counts, args.format
    )
    generated.extend(
        plotting.save_overview_plots(
            args.output_dir, models, prompt_types, counts, args.format
        )
    )
    summary_path = args.output_dir / "parallel_judge_metrics_summary.csv"
    plotting.write_summary_csv(summary_path, models, prompt_types, counts)

    print(f"Generated {len(generated)} plots in {args.output_dir}")
    for path in generated:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
