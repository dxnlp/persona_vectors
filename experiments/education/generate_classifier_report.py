"""
Generate analysis report for the LLM-based student type classifier experiment.

Produces metrics at 4 granularity levels, confusion matrices, and per-trait
detectability analysis.

Usage:
    python -m experiments.education.generate_classifier_report
    python -m experiments.education.generate_classifier_report --input-dir results/classifier
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

ALL_TRAITS = ["evil", "apathetic", "hallucinating", "humorous", "impolite", "optimistic", "sycophantic"]

TRAIT_OPPOSITES = {
    "evil": "good", "apathetic": "empathetic", "hallucinating": "factual",
    "humorous": "serious", "impolite": "polite", "optimistic": "pessimistic",
    "sycophantic": "candid",
}

STUDENT_TYPES_15 = [
    "unsteered",
    "evil_pos", "evil_neg",
    "apathetic_pos", "apathetic_neg",
    "hallucinating_pos", "hallucinating_neg",
    "humorous_pos", "humorous_neg",
    "impolite_pos", "impolite_neg",
    "optimistic_pos", "optimistic_neg",
    "sycophantic_pos", "sycophantic_neg",
]

TRAIT_FAMILIES_8 = ["unsteered"] + ALL_TRAITS

DIRECTIONS_3 = ["unsteered", "positive", "negative"]

MODEL_COLORS = {
    "Qwen3-4B": "#1f77b4",
    "Qwen3-32B": "#ff7f0e",
    "gpt-oss-20b": "#2ca02c",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_classifications(input_dir):
    """Load all classification JSONL files, return DataFrame."""
    all_records = []
    for path in sorted(input_dir.glob("*_classifications.jsonl")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    if not all_records:
        raise FileNotFoundError(f"No classification files found in {input_dir}")
    df = pd.DataFrame(all_records)
    # Filter out errors
    valid = ~df["predicted_type"].str.startswith(("PARSE_ERROR", "API_ERROR", "INVALID:"), na=False)
    n_errors = (~valid).sum()
    if n_errors > 0:
        print(f"  Filtered {n_errors} error rows ({n_errors / len(df) * 100:.1f}%)")
    return df[valid].copy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df, true_col, pred_col, labels=None):
    """Compute accuracy and macro F1 for a pair of columns."""
    y_true = df[true_col]
    y_pred = df[pred_col]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return acc, f1


def compute_all_metrics(df):
    """Compute metrics at all 4 granularity levels."""
    results = {}

    # 15-class
    acc, f1 = compute_metrics(df, "student_type", "predicted_type", labels=STUDENT_TYPES_15)
    results["15-class"] = {"accuracy": acc, "macro_f1": f1}

    # 8-class (trait family)
    acc, f1 = compute_metrics(df, "trait_family_true", "trait_family_pred", labels=TRAIT_FAMILIES_8)
    results["8-class"] = {"accuracy": acc, "macro_f1": f1}

    # 3-class (direction)
    acc, f1 = compute_metrics(df, "direction_true", "direction_pred", labels=DIRECTIONS_3)
    results["3-class"] = {"accuracy": acc, "macro_f1": f1}

    # Binary (steered vs unsteered)
    acc, f1 = compute_metrics(df, "is_steered_true", "is_steered_pred")
    results["binary"] = {"accuracy": acc, "macro_f1": f1}

    return results


def per_trait_f1(df):
    """Compute per-trait (8-class family) F1 scores."""
    results = {}
    for trait in TRAIT_FAMILIES_8:
        sub = df.copy()
        sub["is_trait_true"] = sub["trait_family_true"] == trait
        sub["is_trait_pred"] = sub["trait_family_pred"] == trait
        tp = ((sub["is_trait_true"]) & (sub["is_trait_pred"])).sum()
        fp = ((~sub["is_trait_true"]) & (sub["is_trait_pred"])).sum()
        fn = ((sub["is_trait_true"]) & (~sub["is_trait_pred"])).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[trait] = {"precision": precision, "recall": recall, "f1": f1}
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plot(plot_dir, name):
    path = plot_dir / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"plots/{name}.png"


def plot_accuracy_by_level(all_metrics, plot_dir):
    """1: Grouped bar chart — accuracy at 4 levels x 3 models."""
    levels = ["15-class", "8-class", "3-class", "binary"]
    baselines = {"15-class": 1/15, "8-class": 1/8, "3-class": 7/15, "binary": 14/15}
    models = list(all_metrics.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, metric_name in enumerate(["accuracy", "macro_f1"]):
        ax = axes[metric_idx]
        x = np.arange(len(levels))
        width = 0.8 / (n_models + 1)

        # Baseline bars
        baseline_vals = [baselines[l] for l in levels]
        ax.bar(x - (n_models) / 2 * width, baseline_vals, width,
               label="Random baseline", color="#cccccc", edgecolor="white")

        for mi, model in enumerate(models):
            vals = [all_metrics[model][l][metric_name] for l in levels]
            offset = (mi - (n_models - 1) / 2 + 1) * width
            ax.bar(x + offset, vals, width, label=model,
                   color=MODEL_COLORS.get(model, "#888888"), alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Classifier {metric_name.replace('_', ' ').title()} by Granularity")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return save_plot(plot_dir, "1_accuracy_by_level")


def plot_confusion_matrix_15(df, model, plot_dir):
    """Confusion matrix for 15-class classification."""
    sub = df[df["source_model"] == model]
    # Only include labels present in data
    present = sorted(set(sub["student_type"].unique()) | set(sub["predicted_type"].unique()))
    labels = [l for l in STUDENT_TYPES_15 if l in present]

    cm = confusion_matrix(sub["student_type"], sub["predicted_type"], labels=labels)
    # Normalize by row
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"15-Class Confusion Matrix — {model}\n(row-normalized)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return save_plot(plot_dir, f"2_cm15_{model}")


def plot_confusion_matrix_8(df, model, plot_dir):
    """Confusion matrix for 8-class (trait family) classification."""
    sub = df[df["source_model"] == model]
    present = sorted(set(sub["trait_family_true"].unique()) | set(sub["trait_family_pred"].unique()))
    labels = [l for l in TRAIT_FAMILIES_8 if l in present]

    cm = confusion_matrix(sub["trait_family_true"], sub["trait_family_pred"], labels=labels)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"8-Class (Trait Family) Confusion Matrix — {model}\n(row-normalized)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    return save_plot(plot_dir, f"3_cm8_{model}")


def plot_per_trait_detectability(all_trait_f1, plot_dir):
    """4: Per-trait F1 detectability across models."""
    models = list(all_trait_f1.keys())
    traits = ALL_TRAITS  # exclude "unsteered" for clarity
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(traits))
    width = 0.8 / n_models

    for mi, model in enumerate(models):
        vals = [all_trait_f1[model].get(t, {}).get("f1", 0) for t in traits]
        offset = (mi - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model,
               color=MODEL_COLORS.get(model, "#888888"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TRAIT_OPPOSITES[t]})" for t in traits], fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Trait Detectability (8-class F1) Across Models")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return save_plot(plot_dir, "4_per_trait_detectability")


def plot_misclassification_patterns(df, plot_dir):
    """5: Most common confusion pairs across all models."""
    # Count all (true, pred) pairs where true != pred
    wrong = df[df["student_type"] != df["predicted_type"]]
    pair_counts = Counter(zip(wrong["student_type"], wrong["predicted_type"]))

    if not pair_counts:
        return None

    # Top 20 confusion pairs
    top_pairs = pair_counts.most_common(20)
    labels = [f"{t} -> {p}" for (t, p), _ in top_pairs]
    counts = [c for _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#d62728", alpha=0.75, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("Top 20 Misclassification Pairs (All Models)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return save_plot(plot_dir, "5_misclassification_patterns")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(df, input_dir, output_dir):
    """Generate full classifier report."""
    report_dir = output_dir
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["source_model"].unique())

    # Load config if available
    config_path = input_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Compute metrics per model
    all_metrics = {}
    all_trait_f1 = {}
    for model in models:
        sub = df[df["source_model"] == model]
        all_metrics[model] = compute_all_metrics(sub)
        all_trait_f1[model] = per_trait_f1(sub)

    # Aggregate metrics
    all_metrics["All Models"] = compute_all_metrics(df)
    all_trait_f1["All Models"] = per_trait_f1(df)

    # ---- Generate plots ----
    print("  Generating plots...")

    # Plot 1: Accuracy by level
    p1 = plot_accuracy_by_level(
        {m: all_metrics[m] for m in models},  # exclude aggregate
        plot_dir,
    )

    # Plots 2-3: Confusion matrices per model
    cm15_plots = {}
    cm8_plots = {}
    for model in models:
        cm15_plots[model] = plot_confusion_matrix_15(df, model, plot_dir)
        cm8_plots[model] = plot_confusion_matrix_8(df, model, plot_dir)

    # Plot 4: Per-trait detectability
    p4 = plot_per_trait_detectability(
        {m: all_trait_f1[m] for m in models},
        plot_dir,
    )

    # Plot 5: Misclassification patterns
    p5 = plot_misclassification_patterns(df, plot_dir)

    # ---- Build markdown report ----
    print("  Writing report...")
    md = []
    md.append("# Student Type Classifier Report\n")
    md.append("> Can an LLM identify which persona steering was applied to a student answer?\n")
    md.append(f"> Classifier: {config.get('classifier_model', 'unknown')}")
    md.append(f"> Models tested: {', '.join(models)}")
    md.append(f"> Total classifications: {len(df)}")
    md.append("")

    # ---- Section 1: Overview ----
    md.append("## 1. Classification Accuracy Overview\n")
    md.append(f"![Accuracy by Level]({p1})\n")

    md.append("### Accuracy and Macro F1 by Granularity Level\n")
    baselines = {"15-class": "6.7%", "8-class": "12.5%", "3-class": "46.7%", "binary": "93.3%"}
    levels = ["15-class", "8-class", "3-class", "binary"]

    header = "| Level | Random Baseline |"
    sep = "|-------|----------------|"
    for model in models + ["All Models"]:
        header += f" {model} Acc | {model} F1 |"
        sep += "------|------|"
    md.append(header)
    md.append(sep)

    for level in levels:
        row = f"| {level} | {baselines[level]} |"
        for model in models + ["All Models"]:
            m = all_metrics[model][level]
            row += f" {m['accuracy']:.1%} | {m['macro_f1']:.1%} |"
        md.append(row)
    md.append("")

    # ---- Section 2: Per-model confusion matrices ----
    md.append("---\n## 2. Confusion Matrices\n")

    md.append("### 2.1 15-Class Confusion Matrices\n")
    for model in models:
        md.append(f"#### {model}\n")
        md.append(f"![CM 15-class {model}]({cm15_plots[model]})\n")

    md.append("### 2.2 8-Class (Trait Family) Confusion Matrices\n")
    for model in models:
        md.append(f"#### {model}\n")
        md.append(f"![CM 8-class {model}]({cm8_plots[model]})\n")

    # ---- Section 3: Per-trait detectability ----
    md.append("---\n## 3. Per-Trait Detectability\n")
    md.append(f"![Per-Trait Detectability]({p4})\n")

    md.append("### Trait-Level Precision / Recall / F1\n")
    header = "| Trait |"
    sep = "|-------|"
    for model in models:
        header += f" {model} P | {model} R | {model} F1 |"
        sep += "------|------|------|"
    md.append(header)
    md.append(sep)

    for trait in TRAIT_FAMILIES_8:
        row = f"| {trait} |"
        for model in models:
            t = all_trait_f1[model].get(trait, {"precision": 0, "recall": 0, "f1": 0})
            row += f" {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} |"
        md.append(row)
    md.append("")

    # ---- Section 4: Misclassification patterns ----
    md.append("---\n## 4. Misclassification Patterns\n")
    if p5:
        md.append(f"![Misclassification Patterns]({p5})\n")

    # Top confusions per model
    for model in models:
        sub = df[(df["source_model"] == model) & (df["student_type"] != df["predicted_type"])]
        if len(sub) == 0:
            continue
        pair_counts = Counter(zip(sub["student_type"], sub["predicted_type"]))
        top5 = pair_counts.most_common(5)
        md.append(f"**{model} — Top 5 confusions:**\n")
        for (true, pred), count in top5:
            md.append(f"- {true} misclassified as {pred}: {count} times")
        md.append("")

    # ---- Section 5: Direction analysis ----
    md.append("---\n## 5. Direction Analysis (pos vs neg within traits)\n")
    md.append("How well can the classifier distinguish positive from negative steering "
              "within the same trait family?\n")

    for model in models:
        sub = df[(df["source_model"] == model) & (df["direction_true"] != "unsteered")]
        if len(sub) == 0:
            continue
        md.append(f"**{model}:**\n")
        for trait in ALL_TRAITS:
            trait_sub = sub[sub["trait_family_true"] == trait]
            if len(trait_sub) == 0:
                continue
            # Among answers from this trait, how often is direction correctly predicted?
            correct_dir = (trait_sub["direction_true"] == trait_sub["direction_pred"]).mean()
            # How often is the trait family correct?
            correct_fam = (trait_sub["trait_family_true"] == trait_sub["trait_family_pred"]).mean()
            md.append(f"- {trait}: family accuracy={correct_fam:.1%}, "
                      f"direction accuracy={correct_dir:.1%}")
        md.append("")

    # ---- Section 6: Key Takeaways ----
    md.append("---\n## 6. Key Takeaways\n")

    # Find easiest/hardest traits
    avg_trait_f1 = {}
    for trait in ALL_TRAITS:
        vals = [all_trait_f1[m].get(trait, {}).get("f1", 0) for m in models]
        avg_trait_f1[trait] = np.mean(vals)
    easiest = max(avg_trait_f1, key=avg_trait_f1.get)
    hardest = min(avg_trait_f1, key=avg_trait_f1.get)

    # Most/least classifiable model
    model_15_acc = {m: all_metrics[m]["15-class"]["accuracy"] for m in models}
    best_model = max(model_15_acc, key=model_15_acc.get)
    worst_model = min(model_15_acc, key=model_15_acc.get)

    md.append(f"1. **Easiest trait to detect:** {easiest} (avg F1={avg_trait_f1[easiest]:.2f})")
    md.append(f"2. **Hardest trait to detect:** {hardest} (avg F1={avg_trait_f1[hardest]:.2f})")
    md.append(f"3. **Most classifiable model:** {best_model} "
              f"(15-class acc={model_15_acc[best_model]:.1%})")
    md.append(f"4. **Least classifiable model:** {worst_model} "
              f"(15-class acc={model_15_acc[worst_model]:.1%})")

    agg = all_metrics["All Models"]
    md.append(f"5. **Aggregate 15-class accuracy:** {agg['15-class']['accuracy']:.1%} "
              f"(vs 6.7% random baseline)")
    md.append(f"6. **Aggregate 8-class accuracy:** {agg['8-class']['accuracy']:.1%} "
              f"(vs 12.5% random baseline)")
    md.append(f"7. **Aggregate binary accuracy:** {agg['binary']['accuracy']:.1%} "
              f"(vs 93.3% majority baseline)")
    md.append("")

    md.append("---\n*Generated from `experiments/education/generate_classifier_report.py`*\n")

    # Write report
    report_path = report_dir / "classifier_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Report: {report_path}")
    print(f"  Plots: {plot_dir}")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate classifier experiment report")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing classification JSONL files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for report (default: input-dir/report)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else RESULTS_ROOT / "classifier"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "report"

    print("=" * 70)
    print("CLASSIFIER REPORT GENERATOR")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    print("Loading classifications...")
    df = load_classifications(input_dir)
    print(f"  {len(df)} valid classifications across "
          f"{df['source_model'].nunique()} models")

    generate_report(df, input_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
