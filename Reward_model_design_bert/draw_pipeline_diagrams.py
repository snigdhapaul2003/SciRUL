"""Create presentation-ready training and testing pipeline diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "navy": "#17324D",
    "blue": "#DCEBFA",
    "blue_edge": "#3977A8",
    "green": "#DDF3E4",
    "green_edge": "#348C59",
    "orange": "#FCE8CF",
    "orange_edge": "#C97922",
    "purple": "#E9E1F6",
    "purple_edge": "#7555A5",
    "gray": "#F1F3F5",
    "gray_edge": "#6C757D",
    "red": "#F9DEDE",
    "red_edge": "#B64B4B",
}


def setup(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.55, title, fontsize=25, fontweight="bold", color=COLORS["navy"])
    ax.text(0.6, 8.18, subtitle, fontsize=12.5, color="#526474")
    return fig, ax


def box(ax, x, y, w, h, title, body="", style="blue", dashed=False, fontsize=11):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        facecolor=COLORS[style], edgecolor=COLORS[f"{style}_edge"],
        linewidth=1.8, linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
            fontsize=fontsize + 1, fontweight="bold", color=COLORS["navy"])
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
                fontsize=fontsize, color="#243746", linespacing=1.3)
    return patch


def arrow(ax, start, end, label="", dashed=False, color="#526474", curve=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.8,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}", shrinkA=4, shrinkB=4,
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.14, label, ha="center", va="bottom", fontsize=10,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def training_diagram(output: Path) -> None:
    fig, ax = setup(
        "Training and calibration pipeline",
        "Nine-fold protocol • cross-paper negative augmentation • joint usage and span learning",
    )
    box(ax, 0.45, 6.50, 2.35, 1.25, "Judged dataset",
        "Claim + paragraph\nUsage label + character spans", "gray", fontsize=9)
    box(ax, 3.35, 6.50, 2.15, 1.25, "Nine-fold split",
        "Split by answer position\nParagraphs remain intact", "blue", fontsize=9)
    arrow(ax, (2.8, 7.12), (3.35, 7.12))

    box(ax, 6.15, 6.65, 2.15, 1.0, "Training set", "7 answer positions", "blue", fontsize=9)
    box(ax, 6.15, 4.75, 2.15, 1.0, "Calibration set", "1 answer position", "purple", fontsize=9)
    box(ax, 6.15, 2.85, 2.15, 1.0, "Sealed test set", "1 answer position", "gray", dashed=True, fontsize=9)
    arrow(ax, (5.5, 7.12), (6.15, 7.15))
    arrow(ax, (5.05, 6.5), (6.15, 5.25), curve=0.12)
    arrow(ax, (4.95, 6.5), (6.15, 3.35), curve=0.18, dashed=True)

    box(ax, 9.0, 6.45, 2.45, 1.4, "Cross-paper negatives",
        "+5 claims per paragraph\nOther-paper claim\nused=0 • no span", "orange", fontsize=9)
    box(ax, 9.0, 4.55, 2.45, 1.4, "Cross-paper negatives",
        "Same sampling rule\nBoth classes represented", "orange", fontsize=9)
    arrow(ax, (8.3, 7.15), (9.0, 7.15))
    arrow(ax, (8.3, 5.25), (9.0, 5.25))

    box(ax, 12.25, 6.35, 3.1, 1.6, "Joint DeBERTa",
        "Input: claim + paragraph\nUsage head + BIO span head\nJoint loss + consistency penalty", "green", fontsize=9)
    arrow(ax, (11.45, 7.15), (12.25, 7.15))
    box(ax, 12.25, 4.45, 3.1, 1.6, "Conformal calibration",
        "Trained DeBERTa probabilities\nClass-conditional thresholds\nused • not used • abstain", "purple", fontsize=9)
    arrow(ax, (11.45, 5.25), (12.25, 5.25))
    arrow(ax, (13.8, 6.35), (13.8, 6.05), dashed=True)

    ax.text(0.65, 1.95, "SAVED FOR PARAGRAPH-ONLY INFERENCE", fontsize=11.5,
            fontweight="bold", color=COLORS["navy"])
    box(ax, 1.1, 0.65, 3.65, 1.15, "Trained DeBERTa",
        "Usage verification + BIO localization", "green", fontsize=9)
    box(ax, 6.15, 0.65, 3.65, 1.15, "Conformal thresholds",
        "Decision sets for used / not used / abstain", "purple", fontsize=9)
    box(ax, 11.2, 0.65, 3.65, 1.15, "Frozen SciNCL index",
        "Embeddings of claims seen in training", "blue", fontsize=9)
    ax.text(8.45, 3.25, "No augmentation or model input",
            fontsize=9.5, color=COLORS["gray_edge"], style="italic")
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def testing_diagram(output: Path) -> None:
    fig, ax = setup(
        "Paragraph-only testing and inference",
        "The gold claim and gold span are not available to the inference pipeline",
    )
    xs = [0.25, 2.85, 5.45, 8.05, 10.65, 13.25]
    width = 2.20
    box(ax, xs[0], 5.95, width, 1.4, "External input",
        "Paragraph only\nNo claim • no gold span", "gray", fontsize=9)
    box(ax, xs[1], 5.95, width, 1.4, "Sentence splitting",
        "Sentence units\nExact character offsets", "blue", fontsize=9)
    box(ax, xs[2], 5.95, width, 1.4, "SciNCL retrieval",
        "Sentence embeddings\nSearch frozen claim index", "blue", fontsize=9)
    box(ax, xs[3], 5.95, width, 1.4, "Top-k claims",
        "Default k = 10\nCosine-similarity ranking", "orange", fontsize=9)
    box(ax, xs[4], 5.95, width, 1.4, "DeBERTa verification",
        "Each claim + paragraph\nUsage probability + BIO spans", "green", fontsize=9)
    box(ax, xs[5], 5.95, width, 1.4, "Conformal decision",
        "used • not used • abstain\nAmbiguous sets abstain", "purple", fontsize=9)
    for index in range(5):
        arrow(ax, (xs[index] + width, 6.65), (xs[index + 1], 6.65))

    # Symmetric evaluation inputs: sealed gold on the left and model output on the right.
    box(ax, 0.25, 3.15, 3.10, 1.45, "Gold test annotations",
        "Expected claim IDs + character spans\nEvaluation only — never inference input", "red", dashed=True, fontsize=9)
    box(ax, 12.65, 3.15, 3.10, 1.45, "Final prediction",
        "Decision + claim ID(s)\nExact character spans + stage trace", "green", fontsize=9)
    arrow(ax, (14.35, 5.95), (14.20, 4.60))

    box(ax, 5.65, 0.75, 4.70, 1.65, "Evaluation join",
        "Retrieval recall@k + claim precision/recall\nDecision accuracy + coverage\nCharacter precision, recall, F1 and IoU", "gray", fontsize=9)
    arrow(ax, (3.35, 3.55), (6.45, 2.40), dashed=True, color=COLORS["red_edge"])
    arrow(ax, (12.65, 3.55), (9.55, 2.40), color=COLORS["green_edge"])
    ax.text(0.45, 5.45, "RETRIEVAL", fontsize=11, fontweight="bold", color=COLORS["blue_edge"])
    ax.text(10.75, 5.45, "VERIFICATION & LOCALIZATION", fontsize=11,
            fontweight="bold", color=COLORS["green_edge"])
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "pipeline_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_diagram(output_dir / "training_pipeline.png")
    testing_diagram(output_dir / "testing_pipeline.png")
    print(f"Saved diagrams to {output_dir}")


if __name__ == "__main__":
    main()
