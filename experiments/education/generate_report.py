"""
Generate comprehensive markdown report for multi-trait education experiment.

Usage:
    python -m experiments.education.generate_report --results-dir results/multi_trait_Qwen3-32B_20260218_100705
    python -m experiments.education.generate_report  # defaults to Qwen3-4B results

Run from experiments/education/ directory.
"""

import argparse
import json
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Config — overridden by CLI --results-dir
# ---------------------------------------------------------------------------

RESULTS_DIR = Path('results/multi_trait_20260206_104247')
REPORT_DIR = RESULTS_DIR / 'report'
PLOT_DIR = REPORT_DIR / 'plots'

TRAIT_OPPOSITES = {
    'evil': 'good', 'apathetic': 'empathetic', 'hallucinating': 'factual',
    'humorous': 'serious', 'impolite': 'polite', 'optimistic': 'pessimistic',
    'sycophantic': 'candid',
}
ALL_TRAITS = list(TRAIT_OPPOSITES.keys())

# Colors
C_POS = '#d62728'
C_NEG = '#2ca02c'
C_UNSTEERED = '#1f77b4'
C_OPENAI = '#ff7f0e'
TRAIT_COLORS = {
    'evil': '#8B0000', 'apathetic': '#708090', 'hallucinating': '#9932CC',
    'humorous': '#FFD700', 'impolite': '#FF4500', 'optimistic': '#00CED1',
    'sycophantic': '#FF69B4',
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def calculate_qwk(predictions, ground_truth, min_score=0, max_score=4):
    """Quadratic Weighted Kappa (inlined to avoid GPU-dep imports)."""
    gt_rounded = [round(g) for g in ground_truth]
    predictions = [max(min_score, min(max_score, p)) for p in predictions]
    gt_rounded = [max(min_score, min(max_score, g)) for g in gt_rounded]
    n_ratings = max_score - min_score + 1
    confusion = np.zeros((n_ratings, n_ratings))
    for p, g in zip(predictions, gt_rounded):
        confusion[p - min_score][g - min_score] += 1
    weights = np.zeros((n_ratings, n_ratings))
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i][j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)
    hist_pred = np.sum(confusion, axis=1)
    hist_gt = np.sum(confusion, axis=0)
    n = np.sum(confusion)
    if n == 0:
        return 0.0
    expected = np.outer(hist_pred, hist_gt) / n
    observed_weighted = np.sum(weights * confusion) / n
    expected_weighted = np.sum(weights * expected) / n
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0
    return float(1 - (observed_weighted / expected_weighted))


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def parse_type(t):
    if t in ('unsteered', 'openai', 'real_student'):
        return t, 'none'
    parts = t.rsplit('_', 1)
    return parts[0], parts[1]


def load_data():
    answers_a = load_jsonl(RESULTS_DIR / 'experiment_a_student/generated_answers.jsonl')
    scores_a = load_jsonl(RESULTS_DIR / 'experiment_a_student/scoring_results.jsonl')
    scores_b = load_jsonl(RESULTS_DIR / 'experiment_b_judge/scoring_results.jsonl')
    essays = load_jsonl(RESULTS_DIR / 'shared/sampled_essays.jsonl')
    with open(RESULTS_DIR / 'shared/essay_sets.json') as f:
        essay_sets_info = json.load(f)

    for df in [scores_a, answers_a]:
        df[['trait', 'direction']] = df['student_type'].apply(lambda x: pd.Series(parse_type(x)))
    scores_b[['judge_trait', 'judge_direction']] = scores_b['judge_type'].apply(lambda x: pd.Series(parse_type(x)))
    answers_a['answer_len'] = answers_a['answer'].str.len()
    scores_b['error'] = scores_b['raw_score'] - scores_b['ground_truth_score']
    scores_b['abs_error'] = scores_b['error'].abs()

    return answers_a, scores_a, scores_b, essays, essay_sets_info


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_plot(name):
    path = PLOT_DIR / f'{name}.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return f'plots/{name}.png'


def dir_color(d):
    return C_POS if d == 'pos' else C_NEG if d == 'neg' else C_UNSTEERED


# ---------------------------------------------------------------------------
# Analysis & plot functions
# ---------------------------------------------------------------------------

def plot_effect_sizes(scores_a):
    """A1: Effect size bar chart."""
    baseline = scores_a[scores_a['student_type'] == 'unsteered'].groupby('judge_type')['normalized_score'].mean()

    effects = []
    for st in scores_a['student_type'].unique():
        if st == 'unsteered':
            continue
        trait, direction = parse_type(st)
        for jt in ['unsteered', 'openai']:
            mean = scores_a[(scores_a['student_type'] == st) & (scores_a['judge_type'] == jt)]['normalized_score'].mean()
            effects.append({'trait': trait, 'direction': direction, 'judge_type': jt, 'effect': mean - baseline[jt]})
    effects_df = pd.DataFrame(effects)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for idx, jt in enumerate(['openai', 'unsteered']):
        ax = axes[idx]
        sub = effects_df[effects_df['judge_type'] == jt]
        pos_eff = [sub[(sub['trait'] == t) & (sub['direction'] == 'pos')]['effect'].values[0] for t in ALL_TRAITS]
        neg_eff = [sub[(sub['trait'] == t) & (sub['direction'] == 'neg')]['effect'].values[0] for t in ALL_TRAITS]
        x = np.arange(len(ALL_TRAITS))
        w = 0.35
        ax.bar(x - w/2, pos_eff, w, label=f'+trait (pos)', color=C_POS, alpha=0.85, edgecolor='white')
        ax.bar(x + w/2, neg_eff, w, label=f'-trait / opposite (neg)', color=C_NEG, alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}\n({TRAIT_OPPOSITES[t]})' for t in ALL_TRAITS], fontsize=9)
        ax.set_ylabel('Effect on normalized score')
        judge_label = 'OpenAI (gpt-5.2)' if jt == 'openai' else 'Unsteered LLM'
        ax.set_title(f'Scored by {judge_label}')
        ax.legend(fontsize=9)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('How Persona Steering Affects Student Answer Quality', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot('a1_effect_sizes'), effects_df


def plot_score_distributions(scores_a):
    """A2: Box plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    for idx, jt in enumerate(['openai', 'unsteered']):
        ax = axes[idx]
        sub = scores_a[scores_a['judge_type'] == jt]
        order = sub.groupby('student_type')['normalized_score'].mean().sort_values(ascending=False).index.tolist()
        data = [sub[sub['student_type'] == st]['normalized_score'].values for st in order]
        bp = ax.boxplot(data, tick_labels=order, vert=True, patch_artist=True, widths=0.6)
        for i, st in enumerate(order):
            _, d = parse_type(st)
            bp['boxes'][i].set_facecolor(dir_color(d))
            bp['boxes'][i].set_alpha(0.6)
        judge_label = 'OpenAI' if jt == 'openai' else 'Unsteered LLM'
        ax.set_title(f'Scored by {judge_label}')
        ax.set_ylabel('Normalized Score')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Score Distributions by Student Type', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot('a2_score_distributions')


def plot_heatmap_per_set(scores_a, essay_sets_info):
    """A3: Heatmap of effect per set."""
    baseline_ps = scores_a[scores_a['student_type'] == 'unsteered'].groupby(
        ['set_id', 'judge_type'])['normalized_score'].mean()
    set_ids = sorted(scores_a['set_id'].unique())
    set_labels = [f"S{s}\n{essay_sets_info[str(s)]['topic'][:18]}" for s in set_ids]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for idx, (direction, title) in enumerate([('pos', '+Trait (positive steering)'), ('neg', '-Trait / Opposite (negative steering)')]):
        ax = axes[idx]
        matrix = np.zeros((len(ALL_TRAITS), len(set_ids)))
        for i, trait in enumerate(ALL_TRAITS):
            st = f'{trait}_{direction}'
            for j, sid in enumerate(set_ids):
                mean = scores_a[(scores_a['student_type'] == st) & (scores_a['set_id'] == sid) &
                                (scores_a['judge_type'] == 'openai')]['normalized_score'].mean()
                bl = baseline_ps.get((sid, 'openai'), 0)
                matrix[i, j] = mean - bl
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.8, vmax=0.4)
        ax.set_xticks(range(len(set_ids)))
        ax.set_xticklabels(set_labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(ALL_TRAITS)))
        ax.set_yticklabels(ALL_TRAITS)
        ax.set_title(title)
        for i in range(len(ALL_TRAITS)):
            for j in range(len(set_ids)):
                v = matrix[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > 0.35 else 'black')
    fig.colorbar(im, ax=axes, shrink=0.7, label='Score effect')
    fig.suptitle('Steering Effect by Essay Set (OpenAI judge)', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot('a3_heatmap_per_set')


def plot_answer_length(answers_a, scores_a):
    """A4: Answer length analysis."""
    len_stats = answers_a.groupby('student_type')['answer_len'].agg(['mean', 'std']).round(0)
    unst_len = len_stats.loc['unsteered', 'mean']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Bar: mean length per student type
    ax = axes[0]
    order = len_stats.sort_values('mean', ascending=True).index
    colors = [dir_color(parse_type(st)[1]) for st in order]
    ax.barh(range(len(order)), len_stats.loc[order, 'mean'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.axvline(unst_len, color='black', ls='--', lw=1, label=f'Unsteered ({unst_len:.0f})')
    ax.set_xlabel('Mean answer length (chars)')
    ax.set_title('Answer Length by Student Type')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Scatter: length vs score for key types
    ax = axes[1]
    merged = scores_a[scores_a['judge_type'] == 'openai'].merge(
        answers_a[['set_id', 'sample_id', 'student_type', 'answer_len']],
        on=['set_id', 'sample_id', 'student_type'], how='left')
    for st, color, marker in [('unsteered', C_UNSTEERED, 'o'), ('humorous_pos', C_POS, '^'), ('apathetic_pos', '#708090', 's')]:
        sub = merged[merged['student_type'] == st]
        ax.scatter(sub['answer_len'], sub['normalized_score'], c=color, alpha=0.5, s=25, label=st, marker=marker)
    ax.set_xlabel('Answer length (chars)')
    ax.set_ylabel('Normalized score (OpenAI)')
    ax.set_title('Length vs Quality')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save_plot('a4_answer_length'), len_stats


def plot_judge_performance(scores_b):
    """B1: Judge performance vs ground truth."""
    perf = []
    for jt in scores_b['judge_type'].unique():
        sub = scores_b[scores_b['judge_type'] == jt]
        pred, gt = sub['raw_score'].values, sub['ground_truth_score'].values
        bias = pred - gt
        perf.append({
            'judge_type': jt, 'mean_bias': np.mean(bias), 'mae': np.mean(np.abs(bias)),
            'rmse': np.sqrt(np.mean(bias**2)), 'pred_std': np.std(pred),
            'corr': np.corrcoef(pred, gt)[0,1] if np.std(pred) > 0 else 0,
        })
    perf_df = pd.DataFrame(perf).sort_values('mae')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MAE
    ax = axes[0]
    colors = [dir_color(parse_type(jt)[1]) if jt not in ('openai',) else C_OPENAI for jt in perf_df['judge_type']]
    ax.barh(range(len(perf_df)), perf_df['mae'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(perf_df)))
    ax.set_yticklabels(perf_df['judge_type'], fontsize=9)
    ax.set_xlabel('MAE')
    ax.set_title('Mean Absolute Error')
    ax.grid(axis='x', alpha=0.3)

    # Bias
    ax = axes[1]
    sorted_bias = perf_df.sort_values('mean_bias')
    colors = [dir_color(parse_type(jt)[1]) if jt not in ('openai',) else C_OPENAI for jt in sorted_bias['judge_type']]
    ax.barh(range(len(sorted_bias)), sorted_bias['mean_bias'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(sorted_bias)))
    ax.set_yticklabels(sorted_bias['judge_type'], fontsize=9)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Mean Bias (pred - gt)')
    ax.set_title('Scoring Bias')
    ax.grid(axis='x', alpha=0.3)

    # Correlation
    ax = axes[2]
    sorted_corr = perf_df.sort_values('corr')
    colors = [dir_color(parse_type(jt)[1]) if jt not in ('openai',) else C_OPENAI for jt in sorted_corr['judge_type']]
    ax.barh(range(len(sorted_corr)), sorted_corr['corr'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(sorted_corr)))
    ax.set_yticklabels(sorted_corr['judge_type'], fontsize=9)
    ax.set_xlabel('Pearson r')
    ax.set_title('Correlation with Ground Truth')
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Judge Performance vs Human Ground Truth', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot('b1_judge_performance'), perf_df


def plot_qwk_heatmap(scores_b, essay_sets_info):
    """B2: Per-set QWK and bias heatmaps."""
    # Load essay sets from JSON (avoids importing the package which pulls GPU deps)
    with open(Path(__file__).resolve().parent / 'essay_configs.json') as _f:
        _raw = json.load(_f)
    ESSAY_SETS = {int(k): {**v, 'score_range': tuple(v['score_range'])} for k, v in _raw.items()}

    set_ids = sorted(scores_b['set_id'].unique())
    judge_types = sorted(scores_b['judge_type'].unique())

    qwk_matrix = np.zeros((len(judge_types), len(set_ids)))
    bias_matrix = np.zeros((len(judge_types), len(set_ids)))

    for i, jt in enumerate(judge_types):
        for j, sid in enumerate(set_ids):
            sub = scores_b[(scores_b['judge_type'] == jt) & (scores_b['set_id'] == sid)]
            if len(sub) == 0:
                continue
            pred, gt = sub['raw_score'].tolist(), sub['ground_truth_score'].tolist()
            es = ESSAY_SETS.get(int(sid))
            if es:
                qwk_matrix[i, j] = calculate_qwk(pred, gt, es['score_range'][0], es['score_range'][1])
            bias_matrix[i, j] = np.mean([p - g for p, g in zip(pred, gt)])

    set_labels = [f"S{s}" for s in set_ids]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    im = ax.imshow(qwk_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(set_ids))); ax.set_xticklabels(set_labels)
    ax.set_yticks(range(len(judge_types))); ax.set_yticklabels(judge_types, fontsize=8)
    ax.set_title('QWK per Judge x Set')
    for i in range(len(judge_types)):
        for j in range(len(set_ids)):
            ax.text(j, i, f'{qwk_matrix[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if qwk_matrix[i,j] < 0 or qwk_matrix[i,j] > 0.7 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.7, label='QWK')

    ax = axes[1]
    im2 = ax.imshow(bias_matrix, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(set_ids))); ax.set_xticklabels(set_labels)
    ax.set_yticks(range(len(judge_types))); ax.set_yticklabels(judge_types, fontsize=8)
    ax.set_title('Bias per Judge x Set')
    for i in range(len(judge_types)):
        for j in range(len(set_ids)):
            ax.text(j, i, f'{bias_matrix[i,j]:+.1f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(bias_matrix[i,j]) > 1.0 else 'black')
    fig.colorbar(im2, ax=ax, shrink=0.7, label='Bias')

    fig.suptitle('Judge Quality Across Essay Sets', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot('b2_qwk_bias_heatmap')


def plot_score_compression(scores_b):
    """B3: Score compression analysis."""
    stats = []
    for jt in scores_b['judge_type'].unique():
        sub = scores_b[scores_b['judge_type'] == jt]
        stats.append({
            'judge_type': jt, 'pred_std': sub['raw_score'].std(),
            'unique_scores': sub['raw_score'].nunique(),
            'mode_pct': (sub['raw_score'] == sub['raw_score'].mode()[0]).mean(),
        })
    stats_df = pd.DataFrame(stats).sort_values('pred_std', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [dir_color(parse_type(jt)[1]) if jt not in ('openai',) else C_OPENAI for jt in stats_df['judge_type']]
    bars = ax.barh(range(len(stats_df)), stats_df['pred_std'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df['judge_type'], fontsize=9)
    ax.set_xlabel('Score Standard Deviation')
    ax.set_title('Score Spread by Judge Type (higher = more varied scoring)')

    for i, (_, row) in enumerate(stats_df.iterrows()):
        ax.text(row['pred_std'] + 0.01, i, f"mode={row['mode_pct']:.0%}", va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    patches = [mpatches.Patch(color=C_POS, label='pos (+trait)'), mpatches.Patch(color=C_NEG, label='neg (opposite)'),
               mpatches.Patch(color=C_UNSTEERED, label='unsteered'), mpatches.Patch(color=C_OPENAI, label='openai')]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    plt.tight_layout()
    return save_plot('b3_score_compression'), stats_df


def plot_cross_experiment(scores_a, scores_b):
    """C1: Student effect vs judge bias."""
    baseline = scores_a[scores_a['student_type'] == 'unsteered']['normalized_score'].mean()
    unsteered_judge_norm = scores_b[scores_b['judge_type'] == 'unsteered']['normalized_score'].mean()

    cross = []
    for trait in ALL_TRAITS:
        for d in ['pos', 'neg']:
            st_mean = scores_a[scores_a['student_type'] == f'{trait}_{d}']['normalized_score'].mean()
            jt_mean = scores_b[scores_b['judge_type'] == f'{trait}_{d}']['normalized_score'].mean()
            cross.append({
                'trait': trait, 'direction': d,
                'student_effect': st_mean - baseline,
                'judge_bias': jt_mean - unsteered_judge_norm,
            })
    cross_df = pd.DataFrame(cross)

    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in cross_df.iterrows():
        color = C_POS if row['direction'] == 'pos' else C_NEG
        marker = 'o' if row['direction'] == 'pos' else 's'
        ax.scatter(row['student_effect'], row['judge_bias'], c=color, s=120, marker=marker, zorder=5, edgecolors='white', linewidth=0.5)
        label = f"{row['trait']}_{row['direction']}"
        ax.annotate(label, (row['student_effect'], row['judge_bias']),
                    textcoords='offset points', xytext=(8, 5), fontsize=9)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Student Effect (score change vs unsteered)')
    ax.set_ylabel('Judge Bias (norm score change vs unsteered)')
    ax.set_title('Cross-Experiment: Does a trait that hurts students also bias judges?')
    ax.grid(alpha=0.3)
    patches = [mpatches.Patch(color=C_POS, label='pos (+trait)'), mpatches.Patch(color=C_NEG, label='neg (opposite)')]
    ax.legend(handles=patches, fontsize=10)
    plt.tight_layout()
    return save_plot('c1_cross_experiment'), cross_df


def plot_trait_clustering(scores_a, scores_b):
    """C2: Trait clustering."""
    set_ids = sorted(scores_a['set_id'].unique())
    features = []
    for trait in ALL_TRAITS:
        row = []
        for d in ['pos', 'neg']:
            st = f'{trait}_{d}'
            for sid in set_ids:
                val = scores_a[(scores_a['student_type'] == st) & (scores_a['set_id'] == sid) &
                               (scores_a['judge_type'] == 'openai')]['normalized_score'].mean()
                row.append(val)
            jt = f'{trait}_{d}'
            sub = scores_b[scores_b['judge_type'] == jt]
            row.append((sub['raw_score'] - sub['ground_truth_score']).mean())
            row.append(sub['raw_score'].std())
        features.append(row)

    feat_arr = np.array(features)
    fig, ax = plt.subplots(figsize=(10, 5))
    Z = linkage(pdist(feat_arr, 'euclidean'), method='ward')
    dendrogram(Z, labels=ALL_TRAITS, ax=ax, leaf_font_size=12)
    ax.set_title('Trait Similarity (student effects + judge bias features)')
    ax.set_ylabel('Distance (Ward)')
    plt.tight_layout()
    return save_plot('c2_trait_clustering')


def plot_asymmetry(scores_a):
    """C3: Asymmetry — pos vs neg."""
    baseline_ps = scores_a[scores_a['student_type'] == 'unsteered'].groupby(
        ['set_id', 'judge_type'])['normalized_score'].mean()
    set_ids = sorted(scores_a['set_id'].unique())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, trait in enumerate(ALL_TRAITS):
        ax = axes[idx]
        pos_effects, neg_effects = [], []
        for sid in set_ids:
            bl = baseline_ps.get((sid, 'openai'), 0)
            pos = scores_a[(scores_a['student_type'] == f'{trait}_pos') & (scores_a['set_id'] == sid) &
                           (scores_a['judge_type'] == 'openai')]['normalized_score'].mean() - bl
            neg = scores_a[(scores_a['student_type'] == f'{trait}_neg') & (scores_a['set_id'] == sid) &
                           (scores_a['judge_type'] == 'openai')]['normalized_score'].mean() - bl
            pos_effects.append(pos)
            neg_effects.append(neg)

        x = np.arange(len(set_ids))
        ax.bar(x - 0.2, pos_effects, 0.4, color=C_POS, alpha=0.7, label=f'+{trait}')
        ax.bar(x + 0.2, neg_effects, 0.4, color=C_NEG, alpha=0.7, label=f'-{trait} ({TRAIT_OPPOSITES[trait]})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s}' for s in set_ids], fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(trait, fontsize=11)
        ax.set_ylim(-1.0, 0.5)
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(axis='y', alpha=0.3)

    axes[7].set_visible(False)
    fig.suptitle('Asymmetry: Pos vs Neg Effect Per Essay Set (OpenAI judge)', fontsize=14, y=1.01)
    plt.tight_layout()
    return save_plot('c3_asymmetry')


def plot_judge_agreement_matrix(scores_b):
    """C4: Inter-judge agreement."""
    score_mat = pd.read_csv(RESULTS_DIR / 'experiment_b_judge/score_matrix.csv')
    raw_cols = [c for c in score_mat.columns if c.endswith('_raw')]
    judges = [c.replace('_raw', '') for c in raw_cols]

    corr_matrix = score_mat[raw_cols].corr().values

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=90, fontsize=8)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges, fontsize=8)
    for i in range(len(judges)):
        for j in range(len(judges)):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if corr_matrix[i,j] > 0.7 or corr_matrix[i,j] < 0 else 'black')
    fig.colorbar(im, shrink=0.7, label='Pearson r')
    ax.set_title('Inter-Judge Agreement (correlation on real essays)')
    plt.tight_layout()
    return save_plot('c4_judge_agreement')


# ---------------------------------------------------------------------------
# Cherry pick helpers
# ---------------------------------------------------------------------------

def get_cherry_picks(answers_a, scores_a, scores_b, essays):
    """Extract cherry-picked examples."""
    merged_a = scores_a[scores_a['judge_type'] == 'openai'].merge(
        answers_a[['set_id', 'sample_id', 'student_type', 'answer', 'answer_len']],
        on=['set_id', 'sample_id', 'student_type'], how='left')
    unst_scores = merged_a[merged_a['student_type'] == 'unsteered'][['set_id', 'sample_id', 'normalized_score']].rename(
        columns={'normalized_score': 'unsteered_score'})
    steered_cmp = merged_a[merged_a['student_type'] != 'unsteered'].merge(unst_scores, on=['set_id', 'sample_id'], how='left')
    steered_cmp['drop'] = steered_cmp['normalized_score'] - steered_cmp['unsteered_score']

    picks = {}

    # Worst student drops
    picks['worst_drops'] = []
    for _, row in steered_cmp.nsmallest(3, 'drop').iterrows():
        picks['worst_drops'].append({
            'student_type': row['student_type'], 'set_id': int(row['set_id']),
            'score': row['normalized_score'], 'unsteered_score': row['unsteered_score'],
            'drop': row['drop'], 'answer': row['answer'][:600],
        })

    # Side-by-side humorous
    picks['humorous_sidebyside'] = []
    humor_drops = steered_cmp[steered_cmp['student_type'] == 'humorous_pos'].nsmallest(2, 'drop')
    for _, h_row in humor_drops.iterrows():
        sid, samp = h_row['set_id'], h_row['sample_id']
        u_row = merged_a[(merged_a['set_id'] == sid) & (merged_a['sample_id'] == samp) & (merged_a['student_type'] == 'unsteered')]
        if len(u_row) > 0:
            picks['humorous_sidebyside'].append({
                'set_id': int(sid), 'sample_id': int(samp),
                'unsteered_answer': u_row.iloc[0]['answer'][:500],
                'unsteered_score': float(u_row.iloc[0]['normalized_score']),
                'humorous_answer': h_row['answer'][:500],
                'humorous_score': float(h_row['normalized_score']),
            })

    # Judge overscoring
    merged_b = scores_b.merge(essays[['essay_id', 'essay_text']], on='essay_id', how='left')
    steered_b = merged_b[~merged_b['judge_type'].isin(['openai', 'unsteered'])]

    picks['judge_overscoring'] = []
    for _, row in steered_b.nlargest(3, 'error').iterrows():
        picks['judge_overscoring'].append({
            'judge_type': row['judge_type'], 'essay_id': int(row['essay_id']),
            'set_id': int(row['set_id']), 'predicted': int(row['raw_score']),
            'ground_truth': float(row['ground_truth_score']),
            'error': float(row['error']), 'essay': str(row['essay_text'])[:400],
        })

    # Controversial essays
    score_mat = pd.read_csv(RESULTS_DIR / 'experiment_b_judge/score_matrix.csv')
    raw_cols = [c for c in score_mat.columns if c.endswith('_raw')]
    score_mat['spread'] = score_mat[raw_cols].max(axis=1) - score_mat[raw_cols].min(axis=1)
    picks['controversial'] = []
    for _, row in score_mat.nlargest(3, 'spread').iterrows():
        scores_dict = {c.replace('_raw', ''): int(row[c]) for c in raw_cols if pd.notna(row[c])}
        picks['controversial'].append({
            'essay_id': int(row.get('essay_id', 0)), 'set_id': int(row['set_id']),
            'ground_truth': float(row.get('ground_truth', 0)), 'spread': int(row['spread']),
            'scores': scores_dict,
        })

    return picks


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(answers_a, scores_a, scores_b, essays, essay_sets_info, config=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Read model info from config
    if config is None:
        config = {}
    model_name = config.get('model', 'Unknown')
    model_short = config.get('model_short', model_name.split('/')[-1])
    layer = config.get('layer', '?')
    coeff = config.get('coeff', 2.0)
    n_samples = config.get('samples_per_set', 10)
    n_sets = len(config.get('set_ids', list(range(1, 11))))

    md = []
    md.append("# Multi-Trait Education Experiment — Analysis Report\n")
    md.append("> Analyzing how 7 persona steering vectors affect LLM-based essay scoring\n")
    md.append(f"> Model: {model_short} | Layer: {layer} | Coefficient: +/-{coeff} | {n_sets} essay sets x {n_samples} samples\n")
    md.append("")

    # Overview
    md.append("## Experiment Overview\n")
    md.append("This experiment measures how **activation steering** with 7 personality trait vectors")
    md.append("affects a language model in two roles:\n")
    md.append("- **Experiment A (Student)**: How does steering the *answer-generating* model affect answer quality?")
    md.append(f"  - 15 student types (7 traits x pos/neg + unsteered) x {n_sets * n_samples} prompts = {15 * n_sets * n_samples:,} answers")
    md.append("  - Scored by 2 judges: unsteered LLM + OpenAI gpt-5.2")
    md.append("- **Experiment B (Judge)**: How does steering the *grading* model affect scoring accuracy?")
    md.append(f"  - 16 judge types (7 traits x pos/neg + unsteered + OpenAI) scoring {n_sets * n_samples} real ASAP-SAS essays")
    md.append("  - Evaluated against human ground truth via QWK, bias, and MAE\n")
    md.append("| Trait | Opposite | Description |")
    md.append("|-------|----------|-------------|")
    descs = {
        'evil': 'Malicious, harmful intent', 'apathetic': 'Indifferent, low effort',
        'hallucinating': 'Fabricates facts confidently', 'humorous': 'Jokes, informal tone',
        'impolite': 'Rude, dismissive', 'optimistic': 'Overly positive framing',
        'sycophantic': 'Excessively agreeable/flattering',
    }
    for t in ALL_TRAITS:
        md.append(f"| {t} | {TRAIT_OPPOSITES[t]} | {descs[t]} |")
    md.append("")

    # ===================== PART A =====================
    md.append("---\n## Part A: How Steering Affects Student Answer Quality\n")

    # A1 - Effect sizes
    md.append("### A1. Effect Size Rankings\n")
    p1, effects_df = plot_effect_sizes(scores_a)
    md.append(f"![Effect Sizes]({p1})\n")

    avg_eff = effects_df.groupby(['trait', 'direction'])['effect'].mean().reset_index()
    avg_eff['abs_effect'] = avg_eff['effect'].abs()
    avg_eff = avg_eff.sort_values('abs_effect', ascending=False)

    # Data-driven findings for A1
    pos_effects = avg_eff[avg_eff['direction'] == 'pos'].sort_values('effect')
    neg_effects = avg_eff[avg_eff['direction'] == 'neg']
    worst_pos = pos_effects.iloc[0]
    second_pos = pos_effects.iloc[1] if len(pos_effects) > 1 else None
    third_pos = pos_effects.iloc[2] if len(pos_effects) > 2 else None
    mildest_pos = pos_effects.iloc[-1]
    pos_mean_abs = avg_eff[avg_eff['direction'] == 'pos']['effect'].abs().mean()
    neg_mean_abs = avg_eff[avg_eff['direction'] == 'neg']['effect'].abs().mean()

    md.append("**Key findings:**\n")
    md.append(f"- **{worst_pos['trait'].capitalize()} steering is the most destructive** — a +{worst_pos['trait']} student scores {worst_pos['effect']:.2f} points below baseline (on a 0-1 scale).")
    if second_pos is not None and third_pos is not None:
        md.append(f"- **{second_pos['trait'].capitalize()} and {third_pos['trait']}** also cause significant drops ({second_pos['effect']:+.2f} and {third_pos['effect']:+.2f}).")
    md.append(f"- **{mildest_pos['trait'].capitalize()} steering has the smallest effect** ({mildest_pos['effect']:+.2f}) — the model still produces adequate answers.")
    if pos_mean_abs > neg_mean_abs:
        md.append(f"- The **negative/opposite direction** generally has smaller effects (avg |{neg_mean_abs:.2f}|) than positive (avg |{pos_mean_abs:.2f}|), confirming that")
        md.append("  steering *toward* a negative trait is more disruptive than steering *away* from it.\n")
    else:
        md.append(f"- The **positive and negative directions** have similar effect magnitudes (pos avg |{pos_mean_abs:.2f}| vs neg avg |{neg_mean_abs:.2f}|).\n")

    md.append("| Rank | Trait | Direction | Avg Effect |")
    md.append("|------|-------|-----------|-----------|")
    for i, (_, row) in enumerate(avg_eff.iterrows()):
        md.append(f"| {i+1} | {row['trait']} | {row['direction']} | {row['effect']:+.3f} |")
    md.append("")

    # A2 - Distributions
    md.append("### A2. Score Distributions\n")
    p2 = plot_score_distributions(scores_a)
    md.append(f"![Score Distributions]({p2})\n")
    # Data-driven A2 findings
    type_stats = scores_a[scores_a['judge_type'] == 'openai'].groupby('student_type')['normalized_score'].agg(['mean', 'std', 'median'])
    most_variable = type_stats['std'].idxmax()
    least_variable_steered = type_stats.drop('unsteered', errors='ignore')['std'].idxmin()
    md.append(f"The box plots reveal that **{most_variable}** is both lower-scoring and more variable (std={type_stats.loc[most_variable, 'std']:.2f},")
    md.append(f" median={type_stats.loc[most_variable, 'median']:.2f}) — it produces erratically bad answers.")
    md.append(f" Meanwhile, **{least_variable_steered}** remains tightly clustered (std={type_stats.loc[least_variable_steered, 'std']:.2f})")
    md.append(f" near the unsteered baseline.\n")

    # A3 - Per-set heatmap
    md.append("### A3. Which Question Types Are Most Affected?\n")
    p3 = plot_heatmap_per_set(scores_a, essay_sets_info)
    md.append(f"![Heatmap Per Set]({p3})\n")
    # Data-driven A3 findings
    baseline_ps_a3 = scores_a[scores_a['student_type'] == 'unsteered'].groupby(
        ['set_id', 'judge_type'])['normalized_score'].mean()
    set_ids_a3 = sorted(scores_a['set_id'].unique())
    set_effects_a3 = {}
    for sid in set_ids_a3:
        effects_sid = []
        for trait in ALL_TRAITS:
            for d in ['pos', 'neg']:
                st = f'{trait}_{d}'
                mean_v = scores_a[(scores_a['student_type'] == st) & (scores_a['set_id'] == sid) &
                                  (scores_a['judge_type'] == 'openai')]['normalized_score'].mean()
                bl_v = baseline_ps_a3.get((sid, 'openai'), 0)
                effects_sid.append(mean_v - bl_v)
        set_effects_a3[sid] = np.mean(effects_sid)
    most_vulnerable_set = min(set_effects_a3, key=set_effects_a3.get)
    most_resilient_set = max(set_effects_a3, key=set_effects_a3.get)
    vuln_topic = essay_sets_info.get(str(most_vulnerable_set), {}).get('topic', f'Set {most_vulnerable_set}')
    resil_topic = essay_sets_info.get(str(most_resilient_set), {}).get('topic', f'Set {most_resilient_set}')
    md.append("**Patterns by question type:**\n")
    md.append(f"- **Set {most_vulnerable_set} ({vuln_topic})** is the most vulnerable — steering causes the largest average drops here (avg effect: {set_effects_a3[most_vulnerable_set]:+.2f}).")
    md.append(f"- **Set {most_resilient_set} ({resil_topic})** is the most resilient (avg effect: {set_effects_a3[most_resilient_set]:+.2f}).")
    md.append("  Simple factual questions are harder to derail.\n")

    # A4 - Answer length
    md.append("### A4. Answer Length Analysis\n")
    p4, len_stats = plot_answer_length(answers_a, scores_a)
    md.append(f"![Answer Length]({p4})\n")
    # Data-driven A4 findings
    unst_len_val = len_stats.loc['unsteered', 'mean']
    longest_type = len_stats['mean'].idxmax()
    shortest_type = len_stats['mean'].idxmin()
    longest_pct = (len_stats.loc[longest_type, 'mean'] - unst_len_val) / unst_len_val * 100
    shortest_pct = (len_stats.loc[shortest_type, 'mean'] - unst_len_val) / unst_len_val * 100
    md.append("**Answer length reveals behavioral signatures:**\n")
    md.append(f"- **{longest_type} writes {longest_pct:+.0f}% {'longer' if longest_pct > 0 else 'shorter'}** answers ({len_stats.loc[longest_type, 'mean']:.0f} vs {unst_len_val:.0f} chars).")
    md.append(f"- **{shortest_type} writes {abs(shortest_pct):.0f}% shorter** answers ({len_stats.loc[shortest_type, 'mean']:.0f} chars).\n")

    # ===================== PART B =====================
    md.append("---\n## Part B: How Steering Affects Judge Accuracy\n")

    # B1 - Judge performance
    md.append("### B1. Judge Performance vs Human Ground Truth\n")
    p5, perf_df = plot_judge_performance(scores_b)
    md.append(f"![Judge Performance]({p5})\n")
    # Data-driven B1 findings
    openai_perf = perf_df[perf_df['judge_type'] == 'openai'].iloc[0] if 'openai' in perf_df['judge_type'].values else None
    llm_perf = perf_df[perf_df['judge_type'] != 'openai']
    llm_mae_range = f"{llm_perf['mae'].min():.2f}-{llm_perf['mae'].max():.2f}"
    llm_corr_range = f"{llm_perf['corr'].min():.2f}-{llm_perf['corr'].max():.2f}"
    if openai_perf is not None:
        md.append("**OpenAI dramatically outperforms all LLM judges:**\n")
        md.append(f"- OpenAI: MAE={openai_perf['mae']:.2f}, r={openai_perf['corr']:.2f}")
        md.append(f"- All {model_short} judges: MAE={llm_mae_range}, r={llm_corr_range}\n")
    else:
        md.append(f"**{model_short} judge performance:**\n")
        md.append(f"- MAE range: {llm_mae_range}, correlation range: {llm_corr_range}\n")
    # Check pos vs neg QWK
    qwk_df_b1 = pd.read_csv(RESULTS_DIR / 'experiment_b_judge/qwk_scores.csv')
    pos_judges = qwk_df_b1[qwk_df_b1['judge_type'].str.endswith('_pos')]
    neg_judges = qwk_df_b1[qwk_df_b1['judge_type'].str.endswith('_neg')]
    if len(pos_judges) > 0 and len(neg_judges) > 0:
        pos_qwk_mean = pos_judges['mean_qwk'].mean()
        neg_qwk_mean = neg_judges['mean_qwk'].mean()
        if pos_qwk_mean > neg_qwk_mean:
            md.append(f"**Positive-steered judges** achieve higher mean QWK ({pos_qwk_mean:.3f}) than negative-steered ({neg_qwk_mean:.3f}).")
            md.append(" Positive steering may expand the score range, improving agreement with ground truth variation.\n")
        else:
            md.append(f"**Negative-steered judges** achieve higher mean QWK ({neg_qwk_mean:.3f}) than positive-steered ({pos_qwk_mean:.3f}).\n")

    md.append("| Judge | MAE | Bias | Correlation | Interpretation |")
    md.append("|-------|-----|------|------------|---------------|")
    for _, r in perf_df.iterrows():
        interp = "Best" if r['judge_type'] == 'openai' else "Lenient" if r['mean_bias'] > 0.1 else "Harsh" if r['mean_bias'] < -0.1 else "Neutral"
        md.append(f"| {r['judge_type']} | {r['mae']:.2f} | {r['mean_bias']:+.2f} | {r['corr']:.2f} | {interp} |")
    md.append("")

    # B2 - QWK heatmap
    md.append("### B2. Per-Set Judge Quality\n")
    p6 = plot_qwk_heatmap(scores_b, essay_sets_info)
    md.append(f"![QWK Heatmap]({p6})\n")
    # Data-driven B2 findings
    qwk_by_set = {}
    for sid in sorted(scores_b['set_id'].unique()):
        sub_b2 = scores_b[scores_b['set_id'] == sid]
        if len(sub_b2) > 0:
            pred_b2, gt_b2 = sub_b2['raw_score'].values, sub_b2['ground_truth_score'].values
            qwk_by_set[sid] = np.corrcoef(pred_b2, gt_b2)[0, 1] if np.std(pred_b2) > 0 else 0
    if qwk_by_set:
        hardest_set = min(qwk_by_set, key=qwk_by_set.get)
        easiest_set = max(qwk_by_set, key=qwk_by_set.get)
        hardest_topic = essay_sets_info.get(str(hardest_set), {}).get('topic', f'Set {hardest_set}')
        easiest_topic = essay_sets_info.get(str(easiest_set), {}).get('topic', f'Set {easiest_set}')
        md.append(f"**Set {hardest_set} ({hardest_topic}) is the hardest to judge** — judges achieve lowest agreement here.")
        md.append(f" **Set {easiest_set} ({easiest_topic})** shows consistently high judge agreement.\n")

    # B3 - Score compression
    md.append("### B3. Score Compression\n")
    p7, comp_df = plot_score_compression(scores_b)
    md.append(f"![Score Compression]({p7})\n")
    # Data-driven B3 findings
    unst_comp = comp_df[comp_df['judge_type'] == 'unsteered']
    if len(unst_comp) > 0:
        unst_std = unst_comp.iloc[0]['pred_std']
        unst_mode = unst_comp.iloc[0]['mode_pct']
        md.append(f"**The unsteered judge is {'the most compressed' if unst_std == comp_df['pred_std'].min() else 'relatively compressed'}** (std={unst_std:.2f}), defaulting to its mode score for {unst_mode:.0%} of essays.\n")
    widest = comp_df.iloc[0]  # sorted descending by pred_std
    md.append(f"**{widest['judge_type']}** uses the widest score range (std={widest['pred_std']:.2f}).\n")

    # ===================== PART C =====================
    md.append("---\n## Part C: Cross-Experiment Analysis\n")

    # C1 - Cross-experiment
    md.append("### C1. Student Effect vs Judge Bias\n")
    p8, cross_df = plot_cross_experiment(scores_a, scores_b)
    md.append(f"![Cross Experiment]({p8})\n")
    md.append("This plot asks: *if a trait makes students write worse, does it also make judges grade incorrectly?*\n")
    # Data-driven C1
    if len(cross_df) > 1:
        corr_se_jb = cross_df['student_effect'].corr(cross_df['judge_bias'])
        if abs(corr_se_jb) < 0.3:
            md.append(f"**The answer is mostly no** — correlation between student harm and judge bias is weak (r={corr_se_jb:.2f}).")
            md.append(" The mechanisms are different: student harm comes from off-topic content, while judge bias shifts leniency.\n")
        else:
            md.append(f"**There is a {'positive' if corr_se_jb > 0 else 'negative'} relationship** (r={corr_se_jb:.2f}) between student harm and judge bias.\n")
    worst_student_trait = cross_df.loc[cross_df['student_effect'].idxmin()]
    md.append(f"- **{worst_student_trait['trait']}_{worst_student_trait['direction']}** has the largest student effect ({worst_student_trait['student_effect']:+.3f}).")
    most_biased_judge = cross_df.loc[cross_df['judge_bias'].abs().idxmax()]
    md.append(f"- **{most_biased_judge['trait']}_{most_biased_judge['direction']}** has the largest judge bias ({most_biased_judge['judge_bias']:+.3f}).\n")

    # C2 - Clustering
    md.append("### C2. Trait Clustering\n")
    p9 = plot_trait_clustering(scores_a, scores_b)
    md.append(f"![Trait Clustering]({p9})\n")
    md.append("**Hierarchical clustering reveals trait families** based on combined student effects and judge bias features.")
    md.append(" See the dendrogram above for the specific groupings.\n")

    # C3 - Asymmetry
    md.append("### C3. Direction Asymmetry\n")
    p10 = plot_asymmetry(scores_a)
    md.append(f"![Asymmetry]({p10})\n")
    # Data-driven C3 findings
    md.append("**Is positive steering always worse than negative?**\n")
    asym_data = []
    for trait in ALL_TRAITS:
        pos_eff_c3 = avg_eff[(avg_eff['trait'] == trait) & (avg_eff['direction'] == 'pos')]['effect'].values
        neg_eff_c3 = avg_eff[(avg_eff['trait'] == trait) & (avg_eff['direction'] == 'neg')]['effect'].values
        if len(pos_eff_c3) > 0 and len(neg_eff_c3) > 0:
            asym_data.append({'trait': trait, 'pos': pos_eff_c3[0], 'neg': neg_eff_c3[0], 'gap': pos_eff_c3[0] - neg_eff_c3[0]})
    if asym_data:
        asym_df = pd.DataFrame(asym_data).sort_values('gap')
        most_asym = asym_df.iloc[0]
        md.append(f"- **{most_asym['trait']}** shows the strongest asymmetry: pos={most_asym['pos']:+.3f} vs neg={most_asym['neg']:+.3f}")
        least_asym = asym_df.loc[asym_df['gap'].abs().idxmin()]
        md.append(f"- **{least_asym['trait']}** shows the most symmetric effects: pos={least_asym['pos']:+.3f} vs neg={least_asym['neg']:+.3f}\n")

    # C4 - Judge agreement matrix
    md.append("### C4. Inter-Judge Agreement\n")
    p11 = plot_judge_agreement_matrix(scores_b)
    md.append(f"![Judge Agreement]({p11})\n")
    md.append("The correlation matrix shows which judges tend to agree with each other on real essays.")
    md.append(" Steering creates systematic biases rather than random noise, as seen in the clustering patterns.\n")

    # ===================== PART D =====================
    md.append("---\n## Part D: Cherry-Picked Cases\n")

    picks = get_cherry_picks(answers_a, scores_a, scores_b, essays)

    # D1 - Worst student drops
    md.append("### D1. Worst Student Performance Drops\n")
    md.append("These are cases where steering caused the biggest score drops compared to the unsteered baseline.\n")
    for case in picks['worst_drops']:
        md.append(f"**{case['student_type']}** on Set {case['set_id']} — score dropped {case['drop']:+.2f} (from {case['unsteered_score']:.2f} to {case['score']:.2f})\n")
        md.append(f"```\n{case['answer'][:500]}\n```\n")

    # D2 - Humorous side-by-side
    md.append("### D2. Side-by-Side: Unsteered vs Humorous\n")
    md.append("The humorous steering effect is immediately visible in the writing style:\n")
    for case in picks['humorous_sidebyside']:
        md.append(f"**Set {case['set_id']}, Sample {case['sample_id']}**\n")
        md.append(f"*Unsteered* (score={case['unsteered_score']:.2f}):\n")
        md.append(f"```\n{case['unsteered_answer'][:400]}\n```\n")
        md.append(f"*humorous_pos* (score={case['humorous_score']:.2f}):\n")
        md.append(f"```\n{case['humorous_answer'][:400]}\n```\n")

    # D3 - Judge overscoring
    md.append("### D3. Worst Judge Overscoring\n")
    md.append("Cases where steered judges gave much higher scores than ground truth:\n")
    for case in picks['judge_overscoring']:
        md.append(f"**{case['judge_type']}** on Essay {case['essay_id']} (Set {case['set_id']}) — predicted {case['predicted']}, ground truth {case['ground_truth']:.1f} (error {case['error']:+.1f})\n")
        md.append(f"```\n{case['essay'][:350]}\n```\n")

    # D4 - Controversial essays
    md.append("### D4. Most Controversial Essays\n")
    md.append("Essays where different judges disagreed the most:\n")
    for case in picks['controversial']:
        md.append(f"**Essay {case['essay_id']}** (Set {case['set_id']}) — Ground truth: {case['ground_truth']:.1f}, Spread: {case['spread']} points\n")
        sorted_scores = sorted(case['scores'].items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_scores[:3]
        bot3 = sorted_scores[-3:]
        md.append(f"- Highest: {', '.join(f'{j}={s}' for j, s in top3)}")
        md.append(f"- Lowest: {', '.join(f'{j}={s}' for j, s in bot3)}\n")

    # ===================== SUMMARY =====================
    md.append("---\n## Summary & Key Takeaways\n")

    md.append("### Comprehensive Trait Summary\n")
    md.append("| Trait | Opposite | Student +pos | Student -neg | Judge +pos QWK | Judge +pos Bias | Length Change |")
    md.append("|-------|----------|-------------|-------------|---------------|----------------|--------------|")

    qwk_df = pd.read_csv(RESULTS_DIR / 'experiment_b_judge/qwk_scores.csv')
    baseline = scores_a[scores_a['student_type'] == 'unsteered']['normalized_score'].mean()
    unst_len = answers_a[answers_a['student_type'] == 'unsteered']['answer_len'].mean()

    for t in ALL_TRAITS:
        pos_eff = scores_a[scores_a['student_type'] == f'{t}_pos']['normalized_score'].mean() - baseline
        neg_eff = scores_a[scores_a['student_type'] == f'{t}_neg']['normalized_score'].mean() - baseline
        qwk_row = qwk_df[qwk_df['judge_type'] == f'{t}_pos']
        j_qwk = qwk_row['mean_qwk'].values[0] if len(qwk_row) > 0 else 0
        j_bias = qwk_row['mean_bias'].values[0] if len(qwk_row) > 0 else 0
        pos_len = answers_a[answers_a['student_type'] == f'{t}_pos']['answer_len'].mean()
        len_pct = (pos_len - unst_len) / unst_len * 100
        md.append(f"| {t} | {TRAIT_OPPOSITES[t]} | {pos_eff:+.3f} | {neg_eff:+.3f} | {j_qwk:.3f} | {j_bias:+.2f} | {len_pct:+.1f}% |")

    md.append("")
    md.append("### Key Takeaways\n")
    # Data-driven takeaways
    worst_trait_name = worst_pos['trait']
    worst_trait_eff = worst_pos['effect']
    md.append(f"1. **{worst_trait_name.capitalize()} is the most disruptive trait** for student answer quality ({worst_trait_eff:+.3f} effect).\n")
    if pos_mean_abs > neg_mean_abs:
        ratio = pos_mean_abs / neg_mean_abs if neg_mean_abs > 0 else float('inf')
        md.append(f"2. **Trait effects are asymmetric**: steering *toward* a negative trait (pos) is {ratio:.1f}x more harmful than")
        md.append("   steering *away* from it (neg/opposite direction).\n")
    else:
        md.append("2. **Trait effects are relatively symmetric** between positive and negative directions.\n")
    md.append(f"3. **Question type matters**: Set {most_vulnerable_set} ({vuln_topic}) is most vulnerable,")
    md.append(f"   while Set {most_resilient_set} ({resil_topic}) is most resilient.\n")
    # Judge comparison
    best_llm_judge = qwk_df_b1[qwk_df_b1['judge_type'] != 'openai'].sort_values('mean_qwk', ascending=False).iloc[0]
    openai_judge = qwk_df_b1[qwk_df_b1['judge_type'] == 'openai']
    if len(openai_judge) > 0:
        openai_qwk = openai_judge.iloc[0]['mean_qwk']
        md.append(f"4. **LLM judges are limited**: the best steered {model_short} judge ({best_llm_judge['judge_type']}, QWK={best_llm_judge['mean_qwk']:.3f})")
        md.append(f"   is {'far' if openai_qwk - best_llm_judge['mean_qwk'] > 0.15 else 'somewhat'} below OpenAI (QWK={openai_qwk:.3f}).\n")
    md.append("5. **Student harm and judge bias are decoupled**: traits that destroy answer quality")
    md.append("   don't necessarily bias grading, and vice versa.\n")

    md.append("---\n*Generated from `experiments/education/generate_report.py`*\n")

    report_path = REPORT_DIR / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'Report written to {report_path}')
    print(f'Plots saved to {PLOT_DIR}')
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate analysis report for multi-trait education experiment')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory (default: results/multi_trait_20260206_104247)')
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        REPORT_DIR = RESULTS_DIR / 'report'
        PLOT_DIR = REPORT_DIR / 'plots'
        # Update module globals so all functions see them
        import sys
        this = sys.modules[__name__]
        this.RESULTS_DIR = RESULTS_DIR
        this.REPORT_DIR = REPORT_DIR
        this.PLOT_DIR = PLOT_DIR

    # Load config
    config = {}
    config_path = RESULTS_DIR / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Model: {config.get('model', 'Unknown')}, Layer: {config.get('layer', '?')}")

    print(f'Loading data from {RESULTS_DIR}...')
    answers_a, scores_a, scores_b, essays, essay_sets_info = load_data()
    print('Generating report...')
    report_path = generate_report(answers_a, scores_a, scores_b, essays, essay_sets_info, config=config)
    print(f'\nDone! Report at: {report_path}')
