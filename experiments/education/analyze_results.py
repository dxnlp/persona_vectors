"""
Results analysis and visualization for education scoring experiments.

Usage:
    python -m experiments.education.analyze_results --results-dir experiments/education/results/TIMESTAMP
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .metrics import (
    calculate_qwk,
    calculate_agreement_stats,
    create_confusion_matrix,
    print_results_summary,
    AgreementStats,
)
from .config import ESSAY_SET_INFO


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(results_path)


def load_full_results(results_path: str) -> pd.DataFrame:
    """Load full results including answers from CSV or JSONL file."""
    path = Path(results_path)
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(results_path)


def display_answer_samples(
    df: pd.DataFrame,
    n_samples: int = 3,
    max_length: int = 500,
) -> None:
    """Display sample answers from the full results.

    Args:
        df: DataFrame with full results including generated_answer column
        n_samples: Number of samples to display
        max_length: Maximum characters to display per answer
    """
    if "generated_answer" not in df.columns:
        print("No generated_answer column found. Use full_results.csv instead.")
        return

    print("\n" + "=" * 80)
    print("SAMPLE ANSWERS")
    print("=" * 80)

    # Get unique student configs
    student_configs = df["student_config"].unique() if "student_config" in df.columns else ["unknown"]

    # Sample essays
    essay_ids = df["essay_id"].unique()[:n_samples]

    for essay_id in essay_ids:
        essay_rows = df[df["essay_id"] == essay_id]
        if len(essay_rows) == 0:
            continue

        first_row = essay_rows.iloc[0]
        print(f"\n{'─' * 80}")
        print(f"ESSAY ID: {essay_id} | SET: {first_row.get('essay_set', 'N/A')} | "
              f"GROUND TRUTH: {first_row.get('ground_truth_score', 'N/A')}")
        print(f"PROMPT: {str(first_row.get('prompt', 'N/A'))[:200]}...")
        print(f"{'─' * 80}")

        for _, row in essay_rows.iterrows():
            student_config = row.get("student_config", "unknown")
            answer = str(row.get("generated_answer", ""))

            # Truncate if too long
            if len(answer) > max_length:
                answer = answer[:max_length] + "..."

            print(f"\n[{student_config.upper()}] Answer:")
            print(f"  {answer}")

            # Show scores from different judges
            score_cols = [c for c in row.index if c.startswith("score_")]
            if score_cols:
                scores = ", ".join([f"{c.replace('score_', '')}: {row[c]}" for c in score_cols if pd.notna(row[c])])
                print(f"  SCORES: {scores}")


def compare_answers_by_steering(
    df: pd.DataFrame,
    essay_id: Optional[int] = None,
) -> Dict[str, Dict]:
    """Compare answers generated with different steering configurations.

    Args:
        df: DataFrame with full results
        essay_id: Specific essay to compare (None = all)

    Returns:
        Dictionary with comparison statistics
    """
    if "generated_answer" not in df.columns:
        return {}

    if essay_id is not None:
        df = df[df["essay_id"] == essay_id]

    comparisons = {}

    student_configs = df["student_config"].unique() if "student_config" in df.columns else []

    for config in student_configs:
        config_df = df[df["student_config"] == config]
        answers = config_df["generated_answer"].dropna()

        if len(answers) == 0:
            continue

        # Calculate answer statistics
        answer_lengths = answers.str.len()

        comparisons[config] = {
            "n_answers": len(answers),
            "avg_length": float(answer_lengths.mean()),
            "min_length": int(answer_lengths.min()),
            "max_length": int(answer_lengths.max()),
            "std_length": float(answer_lengths.std()) if len(answers) > 1 else 0,
        }

    return comparisons


def analyze_answer_quality(df: pd.DataFrame) -> None:
    """Analyze and print answer quality metrics by steering configuration."""
    if "generated_answer" not in df.columns:
        print("No generated_answer column found.")
        return

    print("\n" + "=" * 80)
    print("ANSWER QUALITY ANALYSIS")
    print("=" * 80)

    comparisons = compare_answers_by_steering(df)

    print(f"\n{'Config':<15} {'N':>6} {'Avg Len':>10} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-" * 60)

    for config, stats in sorted(comparisons.items()):
        print(f"{config:<15} {stats['n_answers']:>6} {stats['avg_length']:>10.1f} "
              f"{stats['min_length']:>8} {stats['max_length']:>8} {stats['std_length']:>8.1f}")

    # Analyze score distribution by steering
    if "student_config" in df.columns:
        print("\n--- SCORE DISTRIBUTION BY STEERING ---")
        score_cols = [c for c in df.columns if c.startswith("score_")]

        for score_col in score_cols:
            judge_name = score_col.replace("score_", "")
            print(f"\nJudge: {judge_name}")

            for config in df["student_config"].unique():
                config_scores = df[df["student_config"] == config][score_col].dropna()
                if len(config_scores) > 0:
                    print(f"  {config}: mean={config_scores.mean():.2f}, "
                          f"std={config_scores.std():.2f}, n={len(config_scores)}")


def analyze_score_confidence(results_df: pd.DataFrame) -> None:
    """Analyze judge confidence based on score probability distributions.

    This only works if the results include reasoning with probability info.
    """
    if "reasoning" not in results_df.columns:
        return

    print("\n" + "=" * 80)
    print("SCORE CONFIDENCE ANALYSIS")
    print("=" * 80)

    # Check if we have probability information in reasoning
    sample_reasoning = results_df["reasoning"].iloc[0] if len(results_df) > 0 else ""
    if "probabilities:" not in str(sample_reasoning).lower():
        print("\nNo probability information found in results.")
        print("(Score probabilities are only available for local LLM judges)")
        return

    print("\nScore probability distributions indicate judge confidence.")
    print("Higher entropy = less confident, Lower entropy = more confident")

    # Parse probabilities from reasoning strings
    import re

    def parse_probs(reasoning: str) -> Optional[Dict[int, float]]:
        """Parse score probabilities from reasoning string."""
        if not reasoning or "probabilities:" not in reasoning.lower():
            return None
        try:
            # Extract probability pairs like "0:12.34%, 1:56.78%"
            matches = re.findall(r'(\d+):(\d+\.?\d*)%', reasoning)
            if matches:
                return {int(s): float(p) / 100 for s, p in matches}
        except:
            pass
        return None

    for judge_config in results_df["judge_config"].unique():
        judge_df = results_df[results_df["judge_config"] == judge_config]
        probs_list = [parse_probs(r) for r in judge_df["reasoning"]]
        valid_probs = [p for p in probs_list if p is not None]

        if not valid_probs:
            continue

        print(f"\nJudge: {judge_config}")

        # Calculate average entropy
        def entropy(probs: Dict[int, float]) -> float:
            import math
            return -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)

        entropies = [entropy(p) for p in valid_probs]
        avg_entropy = np.mean(entropies)

        # Calculate average max probability (confidence)
        max_probs = [max(p.values()) for p in valid_probs]
        avg_max_prob = np.mean(max_probs)

        print(f"  Average entropy: {avg_entropy:.3f} (lower = more confident)")
        print(f"  Average max prob: {avg_max_prob:.1%} (higher = more confident)")
        print(f"  N samples with probs: {len(valid_probs)}")


def create_results_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a student x judge results matrix.

    Returns:
        DataFrame with student configs as rows, judge configs as columns
    """
    # Get unique configurations
    student_configs = sorted(df["student_config"].unique())
    judge_configs = sorted(df["judge_config"].unique())

    # Calculate QWK for each combination
    matrix_data = []
    for student in student_configs:
        row = {"student_config": student}
        for judge in judge_configs:
            subset = df[(df["student_config"] == student) & (df["judge_config"] == judge)]
            if len(subset) > 0:
                qwk = calculate_qwk(
                    subset["predicted_score"].tolist(),
                    subset["ground_truth_score"].tolist(),
                )
                row[judge] = qwk
            else:
                row[judge] = None
        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index("student_config")


def analyze_bias(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze scoring bias by configuration.

    Returns:
        Dictionary with bias statistics for each configuration
    """
    bias_stats = {}

    for config_pair in df.groupby(["student_config", "judge_config"]).groups.keys():
        student, judge = config_pair
        subset = df[(df["student_config"] == student) & (df["judge_config"] == judge)]

        if len(subset) == 0:
            continue

        predictions = subset["predicted_score"]
        ground_truth = subset["ground_truth_score"]

        # Calculate bias metrics
        bias = (predictions - ground_truth).mean()
        abs_bias = (predictions - ground_truth).abs().mean()

        # Score distribution
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        gt_mean = ground_truth.mean()

        key = f"{student}→{judge}"
        bias_stats[key] = {
            "bias": float(bias),
            "abs_bias": float(abs_bias),
            "pred_mean": float(pred_mean),
            "pred_std": float(pred_std),
            "gt_mean": float(gt_mean),
            "n_samples": len(subset),
            "harsh": bias < -0.5,  # Tends to give lower scores
            "lenient": bias > 0.5,  # Tends to give higher scores
        }

    return bias_stats


def analyze_cross_judge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze agreement between different judges on the same student answers.

    Returns:
        DataFrame with judge-pair agreement statistics
    """
    judge_configs = sorted(df["judge_config"].unique())
    student_configs = sorted(df["student_config"].unique())

    agreement_data = []

    for student in student_configs:
        student_df = df[df["student_config"] == student]

        for i, judge1 in enumerate(judge_configs):
            for judge2 in judge_configs[i+1:]:
                j1_df = student_df[student_df["judge_config"] == judge1]
                j2_df = student_df[student_df["judge_config"] == judge2]

                # Merge on essay_id
                merged = j1_df.merge(
                    j2_df,
                    on="essay_id",
                    suffixes=("_j1", "_j2"),
                )

                if len(merged) > 0:
                    qwk = calculate_qwk(
                        merged["predicted_score_j1"].tolist(),
                        merged["predicted_score_j2"].tolist(),
                    )
                    agreement_data.append({
                        "student": student,
                        "judge1": judge1,
                        "judge2": judge2,
                        "qwk": qwk,
                        "n_samples": len(merged),
                    })

    return pd.DataFrame(agreement_data)


def print_detailed_analysis(df: pd.DataFrame) -> None:
    """Print detailed analysis of results."""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # 1. Results matrix
    print("\n--- QWK RESULTS MATRIX (Student → Judge) ---")
    matrix = create_results_matrix(df)
    print(matrix.to_string())

    # 2. Bias analysis
    print("\n--- BIAS ANALYSIS ---")
    bias_stats = analyze_bias(df)
    print(f"\n{'Configuration':<25} {'Bias':>8} {'Type':>10}")
    print("-" * 50)
    for key, stats in sorted(bias_stats.items()):
        bias_type = "HARSH" if stats["harsh"] else ("LENIENT" if stats["lenient"] else "neutral")
        print(f"{key:<25} {stats['bias']:>+8.3f} {bias_type:>10}")

    # 3. Cross-judge agreement
    print("\n--- CROSS-JUDGE AGREEMENT ---")
    agreement_df = analyze_cross_judge_agreement(df)
    if len(agreement_df) > 0:
        print(agreement_df.to_string(index=False))

    # 4. Per-essay-set analysis
    print("\n--- PER-ESSAY-SET ANALYSIS ---")
    for essay_set in sorted(df["essay_set"].unique()):
        subset = df[df["essay_set"] == essay_set]
        info = ESSAY_SET_INFO.get(essay_set, {})
        print(f"\nEssay Set {essay_set}: {info.get('topic', 'Unknown')}")
        print(f"  Score range: {info.get('score_range', (0, 4))}")
        print(f"  N samples: {len(subset)}")

        for config in subset.groupby(["student_config", "judge_config"]).groups.keys():
            student, judge = config
            config_subset = subset[
                (subset["student_config"] == student) &
                (subset["judge_config"] == judge)
            ]
            if len(config_subset) > 0:
                qwk = calculate_qwk(
                    config_subset["predicted_score"].tolist(),
                    config_subset["ground_truth_score"].tolist(),
                )
                print(f"    {student}→{judge}: QWK={qwk:.3f}")


def generate_report(results_dir: str, output_path: Optional[str] = None) -> str:
    """Generate a markdown report of the experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_path: Path to save the report (default: results_dir/report.md)

    Returns:
        Path to the generated report
    """
    results_dir = Path(results_dir)
    if output_path is None:
        output_path = results_dir / "report.md"

    # Load data
    results_df = load_results(results_dir / "results.csv")

    with open(results_dir / "summary.json") as f:
        summary = json.load(f)

    with open(results_dir / "config.yaml") as f:
        config_text = f.read()

    # Build report
    lines = [
        "# Persona Vectors Education Scoring Experiment Report",
        "",
        f"**Timestamp:** {summary['timestamp']}",
        f"**Model:** {summary['config']['model_name']}",
        f"**Essays:** {summary['n_essays']}",
        f"**Total Results:** {summary['n_results']}",
        "",
        "## Configuration",
        "```yaml",
        config_text,
        "```",
        "",
        "## Results Summary",
        "",
        "### QWK Matrix",
        "",
    ]

    # Add matrix
    matrix = create_results_matrix(results_df)
    lines.append(matrix.to_markdown())
    lines.append("")

    # Add statistics
    lines.append("### Detailed Statistics")
    lines.append("")
    lines.append("| Configuration | QWK | Kappa | MAE | Correlation |")
    lines.append("|--------------|-----|-------|-----|-------------|")

    for config, stats in sorted(summary["stats"].items()):
        lines.append(
            f"| {config} | {stats['qwk']:.3f} | {stats['kappa']:.3f} | "
            f"{stats['mae']:.3f} | {stats['correlation']:.3f} |"
        )

    lines.append("")

    # Add bias analysis
    lines.append("### Bias Analysis")
    lines.append("")
    bias_stats = analyze_bias(results_df)
    lines.append("| Configuration | Bias | Type |")
    lines.append("|--------------|------|------|")
    for key, stats in sorted(bias_stats.items()):
        bias_type = "HARSH" if stats["harsh"] else ("LENIENT" if stats["lenient"] else "neutral")
        lines.append(f"| {key} | {stats['bias']:+.3f} | {bias_type} |")

    lines.append("")

    # Add answer quality analysis if full results available
    full_results_path = results_dir / "full_results.csv"
    if full_results_path.exists():
        full_df = load_full_results(str(full_results_path))
        comparisons = compare_answers_by_steering(full_df)

        lines.append("### Answer Quality by Steering")
        lines.append("")
        lines.append("| Config | N | Avg Length | Min | Max |")
        lines.append("|--------|---|------------|-----|-----|")
        for config, stats in sorted(comparisons.items()):
            lines.append(f"| {config} | {stats['n_answers']} | {stats['avg_length']:.0f} | "
                        f"{stats['min_length']} | {stats['max_length']} |")
        lines.append("")

        # Add sample answers
        lines.append("### Sample Answers")
        lines.append("")

        essay_ids = full_df["essay_id"].unique()[:2]  # First 2 essays
        for essay_id in essay_ids:
            essay_rows = full_df[full_df["essay_id"] == essay_id]
            if len(essay_rows) == 0:
                continue

            first_row = essay_rows.iloc[0]
            lines.append(f"#### Essay {essay_id} (Set {first_row.get('essay_set', 'N/A')}, "
                        f"Ground Truth: {first_row.get('ground_truth_score', 'N/A')})")
            lines.append("")
            lines.append(f"**Prompt:** {str(first_row.get('prompt', 'N/A'))[:300]}...")
            lines.append("")

            for _, row in essay_rows.iterrows():
                student_config = row.get("student_config", "unknown")
                answer = str(row.get("generated_answer", ""))[:500]
                if len(str(row.get("generated_answer", ""))) > 500:
                    answer += "..."

                lines.append(f"**{student_config.upper()} Student:**")
                lines.append(f"> {answer}")
                lines.append("")

                # Show scores
                score_cols = [c for c in row.index if c.startswith("score_")]
                if score_cols:
                    scores = ", ".join([f"{c.replace('score_', '')}: {row[c]}"
                                       for c in score_cols if pd.notna(row[c])])
                    lines.append(f"*Scores: {scores}*")
                    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("*Add your analysis here*")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return str(output_path)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze persona vectors education experiment results"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--generate-report", action="store_true",
        help="Generate markdown report"
    )
    parser.add_argument(
        "--show-answers", action="store_true",
        help="Display sample answers"
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of sample answers to display"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for report"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load and analyze results
    df = load_results(results_dir / "results.csv")

    # Calculate stats
    from .metrics import analyze_results
    stats = analyze_results(
        [
            type("ScoringResult", (), {
                "essay_id": row["essay_id"],
                "essay_set": row["essay_set"],
                "student_config": row["student_config"],
                "judge_config": row["judge_config"],
                "predicted_score": row["predicted_score"],
                "ground_truth_score": row["ground_truth_score"],
            })()
            for _, row in df.iterrows()
        ],
        by_config=True,
    )

    # Print summary
    print_results_summary(stats)

    # Print detailed analysis
    print_detailed_analysis(df)

    # Analyze score confidence (from probability distributions)
    analyze_score_confidence(df)

    # Load and analyze full results if available
    full_results_path = results_dir / "full_results.csv"
    if full_results_path.exists():
        full_df = load_full_results(str(full_results_path))
        analyze_answer_quality(full_df)

        # Display sample answers if requested
        if args.show_answers:
            display_answer_samples(full_df, n_samples=args.n_samples)
    else:
        print("\nNote: full_results.csv not found. Run with newer experiment to get answer analysis.")

    # Generate report if requested
    if args.generate_report:
        generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
