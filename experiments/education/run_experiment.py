"""
Main experiment runner for education scoring research.

Usage:
    # Quick test (1 essay)
    python -m experiments.education.run_experiment --test

    # Full experiment
    python -m experiments.education.run_experiment --essays 20

    # With Claude API judge
    python -m experiments.education.run_experiment --essays 20 --use-api-judge
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import torch

from config_cpu import print_device_info, get_device_map, get_dtype
from eval.model_utils import load_model

from .config import ExperimentConfig, SteeringConfig, ESSAY_SET_INFO
from .data_loader import ASAPDataLoader, Essay, create_sample_dataset
from .student import StudentGenerator, GeneratedAnswer
from .judge import LocalJudge, ClaudeAPIJudge, ScoringResult
from .metrics import analyze_results, print_results_summary, calculate_qwk, calculate_agreement_stats


def load_model_and_tokenizer(config: ExperimentConfig):
    """Load the model and tokenizer."""
    print(f"\nLoading model: {config.model_name}")
    print_device_info()

    model, tokenizer = load_model(config.model_name)
    return model, tokenizer


def create_steering_configs(config: ExperimentConfig) -> List[SteeringConfig]:
    """Create steering configurations for the experiment.

    Returns:
        List of [good, evil, unsteered] configurations
    """
    vector_path = config.get_vector_path("response_avg")

    return [
        SteeringConfig.good(config.layer, config.steering_coef, vector_path),
        SteeringConfig.evil(config.layer, config.steering_coef, vector_path),
        SteeringConfig.unsteered(),
    ]


def generate_all_answers(
    essays: List[Essay],
    model,
    tokenizer,
    steering_configs: List[SteeringConfig],
    config: ExperimentConfig,
) -> Dict[str, List[GeneratedAnswer]]:
    """Generate answers for all steering configurations.

    Returns:
        Dictionary mapping steering config name to list of generated answers
    """
    all_answers = {}

    for steering in steering_configs:
        print(f"\n--- Generating with {steering.name} steering ---")

        generator = StudentGenerator(
            model=model,
            tokenizer=tokenizer,
            steering_config=steering,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        answers = generator.generate_answers(essays, batch_size=1)
        all_answers[steering.name] = answers

        # Save intermediate results
        answers_path = Path(config.output_dir) / f"answers_{steering.name}.jsonl"
        save_answers(answers, answers_path)
        print(f"Saved {len(answers)} answers to {answers_path}")

    return all_answers


def score_all_answers(
    all_answers: Dict[str, List[GeneratedAnswer]],
    model,
    tokenizer,
    steering_configs: List[SteeringConfig],
    config: ExperimentConfig,
) -> List[ScoringResult]:
    """Score all generated answers with all judge configurations.

    Returns:
        List of all scoring results
    """
    all_results = []

    # Local judges (good, evil, unsteered)
    for judge_steering in steering_configs:
        print(f"\n--- Scoring with {judge_steering.name} judge ---")

        judge = LocalJudge(
            model=model,
            tokenizer=tokenizer,
            steering_config=judge_steering,
        )

        for student_config, answers in all_answers.items():
            print(f"  Scoring {student_config} student answers...")
            results = judge.score_batch(answers)
            all_results.extend(results)

    # Claude API judge (optional)
    if config.use_api_judge:
        try:
            print("\n--- Scoring with Claude API judge ---")
            api_judge = ClaudeAPIJudge(
                api_key=config.anthropic_api_key,
                model=config.api_model,
            )

            for student_config, answers in all_answers.items():
                print(f"  Scoring {student_config} student answers...")
                results = api_judge.score_batch(answers)
                all_results.extend(results)
        except Exception as e:
            print(f"Claude API judge failed: {e}")
            print("Continuing without API judge...")

    return all_results


def evaluate_judges_on_original_essays(
    essays: List[Essay],
    model,
    tokenizer,
    steering_configs: List[SteeringConfig],
    config: ExperimentConfig,
) -> Dict[str, Dict]:
    """Evaluate judge performance by scoring ORIGINAL essays and calculating QWK.

    This provides an objective metric for judge quality by comparing
    predicted scores against human ground truth scores.

    Args:
        essays: Original essays from ASAP-SAS with ground truth scores
        model: The model for local judges
        tokenizer: The tokenizer
        steering_configs: List of steering configurations to evaluate
        config: Experiment configuration

    Returns:
        Dictionary with QWK and other metrics for each judge configuration
    """
    print("\n" + "=" * 60)
    print("EVALUATING JUDGES ON ORIGINAL ESSAYS (QWK Calculation)")
    print("=" * 60)

    all_results = []
    judge_metrics = {}

    # Evaluate local judges with different steering
    for steering in steering_configs:
        print(f"\n--- Evaluating {steering.name} judge on original essays ---")

        judge = LocalJudge(
            model=model,
            tokenizer=tokenizer,
            steering_config=steering,
            generate_feedback=False,  # Skip feedback for speed
        )

        results = judge.score_essays(essays, generate_feedback=False)
        all_results.extend(results)

        # Calculate metrics for this judge
        predictions = [r.predicted_score for r in results]
        ground_truth = [r.ground_truth_score for r in results]

        # Get score range from first essay
        if essays:
            min_score, max_score = ESSAY_SET_INFO.get(essays[0].essay_set, {}).get("score_range", (0, 4))
        else:
            min_score, max_score = 0, 4

        qwk = calculate_qwk(predictions, ground_truth, min_score, max_score)
        stats = calculate_agreement_stats(predictions, ground_truth, min_score, max_score)

        judge_metrics[steering.name] = {
            "qwk": qwk,
            "kappa": stats.cohens_kappa,
            "mae": stats.mean_absolute_error,
            "rmse": stats.root_mean_squared_error,
            "correlation": stats.correlation,
            "exact_match": stats.exact_match_rate,
            "within_one": stats.within_one_rate,
            "n_essays": len(results),
        }

        print(f"  QWK: {qwk:.4f}")
        print(f"  MAE: {stats.mean_absolute_error:.4f}")
        print(f"  Exact Match: {stats.exact_match_rate:.1%}")

    # Evaluate Claude API judge if enabled
    if config.use_api_judge:
        try:
            print("\n--- Evaluating Claude API judge on original essays ---")
            api_judge = ClaudeAPIJudge(
                api_key=config.anthropic_api_key,
                model=config.api_model,
                generate_feedback=False,
            )

            results = api_judge.score_essays(essays, generate_feedback=False)
            all_results.extend(results)

            predictions = [r.predicted_score for r in results]
            ground_truth = [r.ground_truth_score for r in results]

            qwk = calculate_qwk(predictions, ground_truth, min_score, max_score)
            stats = calculate_agreement_stats(predictions, ground_truth, min_score, max_score)

            judge_metrics[f"api_{config.api_model}"] = {
                "qwk": qwk,
                "kappa": stats.cohens_kappa,
                "mae": stats.mean_absolute_error,
                "rmse": stats.root_mean_squared_error,
                "correlation": stats.correlation,
                "exact_match": stats.exact_match_rate,
                "within_one": stats.within_one_rate,
                "n_essays": len(results),
            }

            print(f"  QWK: {qwk:.4f}")
            print(f"  MAE: {stats.mean_absolute_error:.4f}")
            print(f"  Exact Match: {stats.exact_match_rate:.1%}")

        except Exception as e:
            print(f"Claude API judge evaluation failed: {e}")

    # Print summary table
    print("\n" + "-" * 60)
    print("JUDGE EVALUATION SUMMARY (Original Essays)")
    print("-" * 60)
    print(f"{'Judge':<20} {'QWK':>8} {'MAE':>8} {'Exact%':>8} {'Within1%':>8}")
    print("-" * 60)
    for judge_name, metrics in sorted(judge_metrics.items()):
        print(f"{judge_name:<20} {metrics['qwk']:>8.4f} {metrics['mae']:>8.4f} "
              f"{metrics['exact_match']*100:>7.1f}% {metrics['within_one']*100:>7.1f}%")
    print("-" * 60)

    return {"results": all_results, "metrics": judge_metrics}


def save_answers(answers: List[GeneratedAnswer], path: Path) -> None:
    """Save generated answers to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for answer in answers:
            f.write(json.dumps({
                "essay_id": answer.essay_id,
                "essay_set": answer.essay_set,
                "prompt": answer.prompt,
                "generated_answer": answer.generated_answer,
                "steering_config": answer.steering_config,
                "ground_truth_score": answer.ground_truth_score,
            }) + "\n")


def save_results(results: List[ScoringResult], path: Path) -> None:
    """Save scoring results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append({
            "essay_id": r.essay_id,
            "essay_set": r.essay_set,
            "student_config": r.student_config,
            "judge_config": r.judge_config,
            "predicted_score": r.predicted_score,
            "ground_truth_score": r.ground_truth_score,
            "reasoning": r.reasoning,
            "quality_feedback": r.quality_feedback,
        })
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Saved {len(results)} results to {path}")


def save_full_results(
    all_answers: Dict[str, List[GeneratedAnswer]],
    all_results: List[ScoringResult],
    path: Path,
) -> None:
    """Save comprehensive results including answers and all scores.

    Creates a single CSV with:
    - Essay metadata
    - Generated answer text for each steering config
    - All judge scores for each answer
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build a comprehensive dataframe
    # First, create answer lookup by (essay_id, student_config)
    answer_lookup = {}
    for student_config, answers in all_answers.items():
        for answer in answers:
            key = (answer.essay_id, student_config)
            answer_lookup[key] = answer

    # Create result lookup by (essay_id, student_config, judge_config)
    result_lookup = {}
    for r in all_results:
        key = (r.essay_id, r.student_config, r.judge_config)
        result_lookup[key] = r

    # Get unique values
    essay_ids = sorted(set(a.essay_id for answers in all_answers.values() for a in answers))
    student_configs = sorted(all_answers.keys())
    judge_configs = sorted(set(r.judge_config for r in all_results))

    # Build rows - one row per (essay_id, student_config)
    rows = []
    for essay_id in essay_ids:
        for student_config in student_configs:
            answer_key = (essay_id, student_config)
            if answer_key not in answer_lookup:
                continue

            answer = answer_lookup[answer_key]
            row = {
                "essay_id": essay_id,
                "essay_set": answer.essay_set,
                "student_config": student_config,
                "prompt": answer.prompt,
                "generated_answer": answer.generated_answer,
                "ground_truth_score": answer.ground_truth_score,
            }

            # Add scores and feedback from each judge
            for judge_config in judge_configs:
                result_key = (essay_id, student_config, judge_config)
                if result_key in result_lookup:
                    result = result_lookup[result_key]
                    row[f"score_{judge_config}"] = result.predicted_score
                    row[f"reasoning_{judge_config}"] = result.reasoning
                    row[f"feedback_{judge_config}"] = result.quality_feedback
                else:
                    row[f"score_{judge_config}"] = None
                    row[f"reasoning_{judge_config}"] = None
                    row[f"feedback_{judge_config}"] = None

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(rows)} comprehensive results to {path}")

    # Also save as JSONL for easier reading of long text fields
    jsonl_path = path.with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved JSONL version to {jsonl_path}")


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the full experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with experiment results and statistics
    """
    print("=" * 80)
    print("PERSONA VECTORS EDUCATION SCORING EXPERIMENT")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"{config.output_dir}/{timestamp}"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_yaml(f"{config.output_dir}/config.yaml")

    # Check for data
    if not Path(config.data_path).exists():
        print(f"\nDataset not found at {config.data_path}")
        print("Creating sample dataset for testing...")
        create_sample_dataset(config.data_path, n_samples=config.sample_size)

    # Load data
    print(f"\nLoading data from {config.data_path}")
    loader = ASAPDataLoader(config.data_path)
    essays = loader.get_essays(
        essay_sets=config.essay_sets,
        sample_size=config.sample_size,
    )
    print(f"Loaded {len(essays)} essays")

    # Check for vectors
    vector_path = config.get_vector_path("response_avg")
    if not Path(vector_path).exists():
        print(f"\nWARNING: Persona vectors not found at {vector_path}")
        print("You need to generate vectors first using generate_vec.py")
        print("Running in unsteered-only mode...")
        steering_configs = [SteeringConfig.unsteered()]
    else:
        steering_configs = create_steering_configs(config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Generate answers
    print("\n" + "=" * 40)
    print("PHASE 1: GENERATING STUDENT ANSWERS")
    print("=" * 40)
    all_answers = generate_all_answers(essays, model, tokenizer, steering_configs, config)

    # Score answers
    print("\n" + "=" * 40)
    print("PHASE 2: SCORING ANSWERS")
    print("=" * 40)
    all_results = score_all_answers(all_answers, model, tokenizer, steering_configs, config)

    # Save results
    results_path = Path(config.output_dir) / "results.csv"
    save_results(all_results, results_path)

    # Save comprehensive results with answers
    full_results_path = Path(config.output_dir) / "full_results.csv"
    save_full_results(all_answers, all_results, full_results_path)

    # Evaluate judges on original essays (QWK calculation)
    print("\n" + "=" * 40)
    print("PHASE 3: EVALUATING JUDGES (QWK)")
    print("=" * 40)
    qwk_evaluation = evaluate_judges_on_original_essays(
        essays, model, tokenizer, steering_configs, config
    )

    # Save QWK evaluation results
    qwk_results_path = Path(config.output_dir) / "qwk_evaluation.csv"
    if qwk_evaluation.get("results"):
        save_results(qwk_evaluation["results"], qwk_results_path)

    # Analyze generated answer results
    print("\n" + "=" * 40)
    print("PHASE 4: ANALYZING GENERATED ANSWER RESULTS")
    print("=" * 40)
    stats = analyze_results(all_results, by_config=True)
    print_results_summary(stats)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "model_name": config.model_name,
            "sample_size": config.sample_size,
            "steering_coef": config.steering_coef,
            "layer": config.layer,
        },
        "n_essays": len(essays),
        "n_results": len(all_results),
        # QWK evaluation on ORIGINAL essays (objective metric)
        "qwk_original_essays": qwk_evaluation.get("metrics", {}),
        # Statistics on generated answer scoring
        "stats_generated_answers": {
            k: {
                "qwk": v.qwk,
                "kappa": v.cohens_kappa,
                "mae": v.mean_absolute_error,
                "correlation": v.correlation,
            }
            for k, v in stats.items()
        },
    }

    summary_path = Path(config.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print final QWK summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS: QWK on Original Essays (Objective Metric)")
    print("=" * 60)
    print("This measures judge quality against human ground truth scores.")
    print("QWK >= 0.70 is considered acceptable for automated scoring.\n")
    qwk_metrics = qwk_evaluation.get("metrics", {})
    if qwk_metrics:
        for judge_name, metrics in sorted(qwk_metrics.items()):
            qwk = metrics.get("qwk", 0)
            status = "✓" if qwk >= 0.70 else "○"
            print(f"  {status} {judge_name}: QWK = {qwk:.4f}")
    else:
        print("  No QWK metrics available.")
    print("=" * 60)

    return summary


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run persona vectors education scoring experiment"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="Model name or path"
    )
    parser.add_argument(
        "--essays", type=int, default=20,
        help="Number of essays to sample per set"
    )
    parser.add_argument(
        "--essay-sets", type=int, nargs="+", default=[1, 2],
        help="Essay set IDs to use"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/asap-sas/train.csv",
        help="Path to ASAP-SAS dataset"
    )
    parser.add_argument(
        "--vector-path", type=str, default=None,
        help="Path to persona vectors directory"
    )
    parser.add_argument(
        "--layer", type=int, default=15,
        help="Layer for steering (Qwen3-4B has 40 layers)"
    )
    parser.add_argument(
        "--coef", type=float, default=2.0,
        help="Steering coefficient"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/education/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use-api-judge", action="store_true",
        help="Use Claude API as additional judge"
    )
    parser.add_argument(
        "--api-model", type=str, default="claude-sonnet-4-20250514",
        help="Claude API model to use"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode (1 essay only)"
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        sample_size=args.essays,
        essay_sets=args.essay_sets,
        data_path=args.data_path,
        vector_path=args.vector_path,
        layer=args.layer,
        steering_coef=args.coef,
        output_dir=args.output_dir,
        use_api_judge=args.use_api_judge,
        api_model=args.api_model,
        test_mode=args.test,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
