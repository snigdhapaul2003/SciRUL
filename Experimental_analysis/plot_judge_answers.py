"""Plot judge-answer metrics by question type and source model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = {
    "retracted_claim_usage": {
        "title": "Retracted claim usage",
        "statuses": ("yes", "incomplete", "no"),
        "colors": {"yes": "#C94C4C", "incomplete": "#E3A72F", "no": "#278C82"},
        "filename": "claim_usage",
    },
    "retraction_explicitly_mentioned": {
        "title": "Retraction explicitly mentioned",
        "statuses": ("yes", "no"),
        "colors": {"yes": "#3975B7", "no": "#8A9099"},
        "filename": "explicit_retraction_mention",
    },
}


def display_model_name(name: str) -> str:
    """Convert internal model identifiers into compact plot labels."""
    replacements = {
        "olmo_3_7B_Instruct": "OLMo-3 7B",
        "llama_3_8B_Instruct": "Llama-3.1 8B",
        "gpt-5.5": "GPT-5.5",
    }
    return replacements.get(name, name.replace("_", " "))


def extract_results(
    document: dict[str, Any],
) -> tuple[list[str], list[int], dict[tuple[int, str, str], Counter[str]]]:
    """Collect status counts indexed by question type, model, and metric."""
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a 'records' list")

    models: list[str] = []
    question_types: set[int] = set()
    counts: dict[tuple[int, str, str], Counter[str]] = defaultdict(Counter)

    for record in records:
        if not isinstance(record, dict):
            continue
        model_outputs = record.get("model_outputs", [])
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
                question_type = int(suffix)
                question_types.add(question_type)

                for metric, settings in METRICS.items():
                    status = answer.get(metric)
                    if status not in settings["statuses"]:
                        raise ValueError(
                            f"Invalid {metric} value {status!r} for {model}, "
                            f"question type {question_type}"
                        )
                    counts[(question_type, model, metric)][status] += 1

    if not models or not question_types:
        raise ValueError("No valid judge_answer_n objects were found")
    return models, sorted(question_types), counts


def plot_metric(
    axis,
    *,
    question_type: int,
    metric: str,
    models: list[str],
    counts: dict[tuple[int, str, str], Counter[str]],
    include_title: bool = True,
) -> None:
    """Draw one normalized stacked-bar comparison on an axis."""
    settings = METRICS[metric]
    x_positions = list(range(len(models)))
    bottoms = [0.0] * len(models)

    for status in settings["statuses"]:
        percentages = []
        raw_counts = []
        for model in models:
            values = counts[(question_type, model, metric)]
            total = sum(values.values())
            count = values[status]
            raw_counts.append(count)
            percentages.append((count / total * 100) if total else 0.0)

        bars = axis.bar(
            x_positions,
            percentages,
            bottom=bottoms,
            width=0.62,
            label=status.capitalize(),
            color=settings["colors"][status],
            edgecolor="white",
            linewidth=0.8,
        )
        for index, (bar, percent, count) in enumerate(
            zip(bars, percentages, raw_counts)
        ):
            if percent >= 5:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[index] + percent / 2,
                    f"{count}\n{percent:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if status != "incomplete" else "#202124",
                    fontweight="semibold",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, percentages)]

    totals = [
        sum(counts[(question_type, model, metric)].values()) for model in models
    ]
    axis.set_xticks(x_positions, [display_model_name(model) for model in models])
    axis.set_ylim(0, 108)
    axis.set_ylabel("Share of answers (%)")
    axis.set_xlabel("Source model")
    if include_title:
        axis.set_title(
            f"Question type {question_type}: {settings['title']}",
            fontsize=13,
            pad=14,
        )
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for index, total in enumerate(totals):
        axis.text(index, 102, f"n={total}", ha="center", va="bottom", fontsize=8)


def save_individual_plots(
    output_dir: Path,
    models: list[str],
    question_types: list[int],
    counts: dict[tuple[int, str, str], Counter[str]],
    file_format: str,
) -> list[Path]:
    """Create one plot for each question-type and metric combination."""
    paths = []
    for question_type in question_types:
        for metric, settings in METRICS.items():
            figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
            plot_metric(
                axis,
                question_type=question_type,
                metric=metric,
                models=models,
                counts=counts,
            )
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=len(settings["statuses"]),
                frameon=False,
            )
            path = output_dir / (
                f"question_type_{question_type}_{settings['filename']}.{file_format}"
            )
            figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def save_overview_plots(
    output_dir: Path,
    models: list[str],
    question_types: list[int],
    counts: dict[tuple[int, str, str], Counter[str]],
    file_format: str,
) -> list[Path]:
    """Create one multi-panel overview for each metric."""
    paths = []
    for metric, settings in METRICS.items():
        figure, axes = plt.subplots(
            1,
            len(question_types),
            figsize=(6.1 * len(question_types), 5.6),
            sharey=True,
            constrained_layout=True,
        )
        if len(question_types) == 1:
            axes = [axes]
        for axis, question_type in zip(axes, question_types):
            plot_metric(
                axis,
                question_type=question_type,
                metric=metric,
                models=models,
                counts=counts,
                include_title=False,
            )
            axis.set_title(f"Question type {question_type}", fontsize=12, pad=12)
        figure.suptitle(settings["title"], fontsize=16, fontweight="semibold")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(settings["statuses"]),
            frameon=False,
        )
        path = output_dir / f"overview_{settings['filename']}.{file_format}"
        figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def write_summary_csv(
    path: Path,
    models: list[str],
    question_types: list[int],
    counts: dict[tuple[int, str, str], Counter[str]],
) -> None:
    """Write the counts and percentages represented in the plots."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "question_type",
                "model",
                "metric",
                "status",
                "count",
                "total",
                "percentage",
            ),
        )
        writer.writeheader()
        for question_type in question_types:
            for model in models:
                for metric, settings in METRICS.items():
                    values = counts[(question_type, model, metric)]
                    total = sum(values.values())
                    for status in settings["statuses"]:
                        count = values[status]
                        writer.writerow(
                            {
                                "question_type": question_type,
                                "model": model,
                                "metric": metric,
                                "status": status,
                                "count": count,
                                "total": total,
                                "percentage": round(count / total * 100, 2)
                                if total
                                else 0,
                            }
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot judge metrics by question type and source model."
    )
    parser.add_argument("--input", type=Path, default=Path("judge_answers.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("plot3"))
    parser.add_argument(
        "--format", choices=("png", "pdf", "svg"), default="png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    models, question_types, counts = extract_results(document)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = save_individual_plots(
        args.output_dir, models, question_types, counts, args.format
    )
    generated.extend(
        save_overview_plots(
            args.output_dir, models, question_types, counts, args.format
        )
    )
    summary_path = args.output_dir / "judge_metrics_summary.csv"
    write_summary_csv(summary_path, models, question_types, counts)

    print(f"Generated {len(generated)} plots in {args.output_dir}")
    for path in generated:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
