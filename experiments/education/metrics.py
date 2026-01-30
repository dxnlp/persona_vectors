"""
Metrics for evaluating scoring agreement.

Includes Quadratic Weighted Kappa (QWK) and other agreement statistics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .judge import ScoringResult


def calculate_qwk(
    predictions: List[int],
    ground_truth: List[float],
    min_score: int = 0,
    max_score: int = 4,
) -> float:
    """Calculate Quadratic Weighted Kappa (QWK).

    QWK measures agreement between predicted and actual scores,
    penalizing larger disagreements more heavily.

    Args:
        predictions: List of predicted scores
        ground_truth: List of ground truth scores
        min_score: Minimum possible score
        max_score: Maximum possible score

    Returns:
        QWK value between -1 and 1 (1 = perfect agreement)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0

    # Round ground truth to integers for comparison
    gt_rounded = [round(g) for g in ground_truth]

    # Clip to valid range
    predictions = [max(min_score, min(max_score, p)) for p in predictions]
    gt_rounded = [max(min_score, min(max_score, g)) for g in gt_rounded]

    n_ratings = max_score - min_score + 1

    # Build confusion matrix
    confusion = np.zeros((n_ratings, n_ratings))
    for p, g in zip(predictions, gt_rounded):
        confusion[p - min_score][g - min_score] += 1

    # Build weight matrix (quadratic weights)
    weights = np.zeros((n_ratings, n_ratings))
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i][j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    # Calculate expected matrix (outer product of marginals)
    hist_pred = np.sum(confusion, axis=1)
    hist_gt = np.sum(confusion, axis=0)
    n = np.sum(confusion)

    if n == 0:
        return 0.0

    expected = np.outer(hist_pred, hist_gt) / n

    # Calculate QWK
    observed_weighted = np.sum(weights * confusion) / n
    expected_weighted = np.sum(weights * expected) / n

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    qwk = 1 - (observed_weighted / expected_weighted)
    return float(qwk)


def calculate_cohens_kappa(
    predictions: List[int],
    ground_truth: List[float],
    min_score: int = 0,
    max_score: int = 4,
) -> float:
    """Calculate Cohen's Kappa (unweighted).

    Args:
        predictions: List of predicted scores
        ground_truth: List of ground truth scores
        min_score: Minimum possible score
        max_score: Maximum possible score

    Returns:
        Kappa value between -1 and 1
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0

    gt_rounded = [round(g) for g in ground_truth]
    predictions = [max(min_score, min(max_score, p)) for p in predictions]
    gt_rounded = [max(min_score, min(max_score, g)) for g in gt_rounded]

    n_ratings = max_score - min_score + 1
    n = len(predictions)

    # Build confusion matrix
    confusion = np.zeros((n_ratings, n_ratings))
    for p, g in zip(predictions, gt_rounded):
        confusion[p - min_score][g - min_score] += 1

    # Observed agreement
    po = np.sum(np.diag(confusion)) / n

    # Expected agreement
    hist_pred = np.sum(confusion, axis=1) / n
    hist_gt = np.sum(confusion, axis=0) / n
    pe = np.sum(hist_pred * hist_gt)

    if pe == 1:
        return 1.0 if po == 1 else 0.0

    kappa = (po - pe) / (1 - pe)
    return float(kappa)


@dataclass
class AgreementStats:
    """Statistics for agreement analysis."""
    n_samples: int
    qwk: float
    cohens_kappa: float
    mean_absolute_error: float
    root_mean_squared_error: float
    correlation: float
    exact_match_rate: float
    within_one_rate: float
    mean_predicted: float
    mean_ground_truth: float
    std_predicted: float
    std_ground_truth: float


def calculate_agreement_stats(
    predictions: List[int],
    ground_truth: List[float],
    min_score: int = 0,
    max_score: int = 4,
) -> AgreementStats:
    """Calculate comprehensive agreement statistics.

    Args:
        predictions: List of predicted scores
        ground_truth: List of ground truth scores
        min_score: Minimum possible score
        max_score: Maximum possible score

    Returns:
        AgreementStats with all metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    n = len(predictions)
    if n == 0:
        return AgreementStats(
            n_samples=0, qwk=0, cohens_kappa=0, mean_absolute_error=0,
            root_mean_squared_error=0, correlation=0, exact_match_rate=0,
            within_one_rate=0, mean_predicted=0, mean_ground_truth=0,
            std_predicted=0, std_ground_truth=0,
        )

    predictions = np.array(predictions, dtype=float)
    ground_truth = np.array(ground_truth, dtype=float)

    # Basic statistics
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))

    # Correlation
    if np.std(predictions) > 0 and np.std(ground_truth) > 0:
        correlation = np.corrcoef(predictions, ground_truth)[0, 1]
    else:
        correlation = 0.0

    # Match rates
    gt_rounded = np.round(ground_truth)
    exact_match = np.mean(predictions == gt_rounded)
    within_one = np.mean(np.abs(predictions - gt_rounded) <= 1)

    return AgreementStats(
        n_samples=n,
        qwk=calculate_qwk(list(predictions.astype(int)), list(ground_truth), min_score, max_score),
        cohens_kappa=calculate_cohens_kappa(list(predictions.astype(int)), list(ground_truth), min_score, max_score),
        mean_absolute_error=float(mae),
        root_mean_squared_error=float(rmse),
        correlation=float(correlation),
        exact_match_rate=float(exact_match),
        within_one_rate=float(within_one),
        mean_predicted=float(np.mean(predictions)),
        mean_ground_truth=float(np.mean(ground_truth)),
        std_predicted=float(np.std(predictions)),
        std_ground_truth=float(np.std(ground_truth)),
    )


def analyze_results(
    results: List[ScoringResult],
    by_config: bool = True,
) -> Dict[str, AgreementStats]:
    """Analyze scoring results by configuration.

    Args:
        results: List of scoring results
        by_config: Whether to group by student-judge configuration pair

    Returns:
        Dictionary mapping configuration names to agreement stats
    """
    from .config import ESSAY_SET_INFO

    # Group results by configuration
    grouped = defaultdict(list)

    for r in results:
        if by_config:
            key = f"{r.student_config}→{r.judge_config}"
        else:
            key = "all"
        grouped[key].append(r)

    # Calculate stats for each group
    stats = {}
    for key, group_results in grouped.items():
        predictions = [r.predicted_score for r in group_results]
        ground_truth = [r.ground_truth_score for r in group_results]

        # Get score range (use first result's essay set)
        if group_results:
            essay_set = group_results[0].essay_set
            info = ESSAY_SET_INFO.get(essay_set, {})
            min_score, max_score = info.get("score_range", (0, 4))
        else:
            min_score, max_score = 0, 4

        stats[key] = calculate_agreement_stats(
            predictions, ground_truth, min_score, max_score
        )

    return stats


def print_results_summary(stats: Dict[str, AgreementStats]) -> None:
    """Print a formatted summary of results.

    Args:
        stats: Dictionary of agreement statistics by configuration
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Configuration':<25} {'QWK':>8} {'Kappa':>8} {'MAE':>8} {'RMSE':>8} {'Corr':>8} {'N':>6}")
    print("-" * 80)

    # Results
    for key, s in sorted(stats.items()):
        print(f"{key:<25} {s.qwk:>8.3f} {s.cohens_kappa:>8.3f} {s.mean_absolute_error:>8.3f} "
              f"{s.root_mean_squared_error:>8.3f} {s.correlation:>8.3f} {s.n_samples:>6}")

    print("-" * 80)

    # Additional details
    print("\nDetailed Statistics:")
    for key, s in sorted(stats.items()):
        print(f"\n  {key}:")
        print(f"    Exact match rate: {s.exact_match_rate:.1%}")
        print(f"    Within 1 point:   {s.within_one_rate:.1%}")
        print(f"    Mean predicted:   {s.mean_predicted:.2f} (std: {s.std_predicted:.2f})")
        print(f"    Mean ground truth:{s.mean_ground_truth:.2f} (std: {s.std_ground_truth:.2f})")


def create_confusion_matrix(
    predictions: List[int],
    ground_truth: List[float],
    min_score: int = 0,
    max_score: int = 4,
) -> np.ndarray:
    """Create a confusion matrix for visualization.

    Args:
        predictions: List of predicted scores
        ground_truth: List of ground truth scores
        min_score: Minimum possible score
        max_score: Maximum possible score

    Returns:
        Confusion matrix as numpy array
    """
    gt_rounded = [round(g) for g in ground_truth]
    predictions = [max(min_score, min(max_score, p)) for p in predictions]
    gt_rounded = [max(min_score, min(max_score, g)) for g in gt_rounded]

    n_ratings = max_score - min_score + 1
    confusion = np.zeros((n_ratings, n_ratings), dtype=int)

    for p, g in zip(predictions, gt_rounded):
        confusion[p - min_score][g - min_score] += 1

    return confusion
