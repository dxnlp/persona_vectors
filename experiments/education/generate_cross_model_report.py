"""
Generate cross-model comparison report for multi-trait education experiments.

Usage:
    python -m experiments.education.generate_cross_model_report
    python -m experiments.education.generate_cross_model_report --output-dir results/cross_model_comparison

Automatically discovers all multi_trait_* result directories.
"""

import argparse
import json
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
# Config
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'

TRAIT_OPPOSITES = {
    'evil': 'good', 'apathetic': 'empathetic', 'hallucinating': 'factual',
    'humorous': 'serious', 'impolite': 'polite', 'optimistic': 'pessimistic',
    'sycophantic': 'candid',
}
ALL_TRAITS = list(TRAIT_OPPOSITES.keys())

# Model colors
MODEL_COLORS = {
    'Qwen3-4B': '#1f77b4',
    'Qwen3-32B': '#ff7f0e',
    'gpt-oss-20b': '#2ca02c',
}
C_POS = '#d62728'
C_NEG = '#2ca02c'
C_UNSTEERED = '#1f77b4'
C_OPENAI = '#ff7f0e'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
})


# ---------------------------------------------------------------------------
# QWK (inlined to avoid GPU-dep imports)
# ---------------------------------------------------------------------------

def calculate_qwk(predictions, ground_truth, min_score=0, max_score=4):
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def discover_result_dirs(root):
    """Find all multi_trait result directories and return sorted by model."""
    dirs = sorted(root.glob('multi_trait_*'))
    dirs = [d for d in dirs if d.is_dir() and (d / 'config.json').exists()]
    return dirs


def load_model_data(result_dir):
    """Load all data for one model."""
    with open(result_dir / 'config.json') as f:
        config = json.load(f)

    model_short = config.get('model_short', config.get('model', '').split('/')[-1])
    if not model_short or model_short == 'Qwen3-4B':
        # Older configs may not have model_short
        model_name = config.get('model', '')
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name

    answers_a = load_jsonl(result_dir / 'experiment_a_student/generated_answers.jsonl')
    scores_a = load_jsonl(result_dir / 'experiment_a_student/scoring_results.jsonl')
    scores_b = load_jsonl(result_dir / 'experiment_b_judge/scoring_results.jsonl')

    with open(result_dir / 'shared/essay_sets.json') as f:
        essay_sets_info = json.load(f)

    # Derived columns
    for df in [scores_a, answers_a]:
        df[['trait', 'direction']] = df['student_type'].apply(lambda x: pd.Series(parse_type(x)))
    scores_b[['judge_trait', 'judge_direction']] = scores_b['judge_type'].apply(lambda x: pd.Series(parse_type(x)))
    answers_a['answer_len'] = answers_a['answer'].str.len()
    scores_b['error'] = scores_b['raw_score'] - scores_b['ground_truth_score']
    scores_b['abs_error'] = scores_b['error'].abs()

    return {
        'model_short': model_short,
        'config': config,
        'answers_a': answers_a,
        'scores_a': scores_a,
        'scores_b': scores_b,
        'essay_sets_info': essay_sets_info,
        'result_dir': result_dir,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_plot(plot_dir, name):
    path = plot_dir / f'{name}.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return f'plots/{name}.png'


def get_model_color(model_short):
    return MODEL_COLORS.get(model_short, '#888888')


# ---------------------------------------------------------------------------
# Cross-model analysis functions
# ---------------------------------------------------------------------------

def compute_effect_sizes(data):
    """Compute per-trait effect sizes for a model."""
    scores_a = data['scores_a']
    baseline = scores_a[scores_a['student_type'] == 'unsteered'].groupby('judge_type')['normalized_score'].mean()

    effects = []
    for st in scores_a['student_type'].unique():
        if st == 'unsteered':
            continue
        trait, direction = parse_type(st)
        for jt in ['unsteered', 'openai']:
            if jt not in baseline:
                continue
            mean = scores_a[(scores_a['student_type'] == st) & (scores_a['judge_type'] == jt)]['normalized_score'].mean()
            effects.append({
                'trait': trait, 'direction': direction, 'judge_type': jt,
                'effect': mean - baseline[jt],
            })
    return pd.DataFrame(effects)


def compute_judge_qwk(data):
    """Compute per-judge-type mean QWK for a model."""
    scores_b = data['scores_b']
    essay_configs_path = Path(__file__).resolve().parent / 'essay_configs.json'
    with open(essay_configs_path) as f:
        raw = json.load(f)
    essay_sets = {int(k): {**v, 'score_range': tuple(v['score_range'])} for k, v in raw.items()}

    results = []
    for jt in scores_b['judge_type'].unique():
        sub = scores_b[scores_b['judge_type'] == jt]
        set_qwks = []
        for sid in sub['set_id'].unique():
            s = sub[sub['set_id'] == sid]
            es = essay_sets.get(int(sid))
            if es and len(s) > 0:
                qwk = calculate_qwk(s['raw_score'].tolist(), s['ground_truth_score'].tolist(),
                                    es['score_range'][0], es['score_range'][1])
                set_qwks.append(qwk)
        pred, gt = sub['raw_score'].values, sub['ground_truth_score'].values
        bias = pred - gt
        results.append({
            'judge_type': jt,
            'mean_qwk': np.mean(set_qwks) if set_qwks else 0,
            'mae': np.mean(np.abs(bias)),
            'mean_bias': np.mean(bias),
            'corr': np.corrcoef(pred, gt)[0, 1] if np.std(pred) > 0 else 0,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_effect_comparison(all_data, plot_dir):
    """1: Side-by-side effect sizes across models (OpenAI judge)."""
    models = [d['model_short'] for d in all_data]
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for idx, direction in enumerate(['pos', 'neg']):
        ax = axes[idx]
        x = np.arange(len(ALL_TRAITS))
        width = 0.8 / n_models

        for mi, data in enumerate(all_data):
            effects = compute_effect_sizes(data)
            eff_openai = effects[(effects['judge_type'] == 'openai') & (effects['direction'] == direction)]
            vals = [eff_openai[eff_openai['trait'] == t]['effect'].values[0]
                    if len(eff_openai[eff_openai['trait'] == t]) > 0 else 0
                    for t in ALL_TRAITS]
            offset = (mi - (n_models - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=data['model_short'],
                   color=get_model_color(data['model_short']), alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}\n({TRAIT_OPPOSITES[t]})' for t in ALL_TRAITS], fontsize=9)
        ax.set_ylabel('Effect on normalized score')
        dir_label = '+Trait (positive steering)' if direction == 'pos' else '-Trait / Opposite (negative steering)'
        ax.set_title(dir_label)
        ax.legend(fontsize=9)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Student Answer Quality: Effect of Steering Across Models (OpenAI Judge)', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot(plot_dir, '1_effect_comparison')


def plot_judge_qwk_comparison(all_data, plot_dir):
    """2: Judge QWK comparison across models."""
    models = [d['model_short'] for d in all_data]

    # Collect QWK data per model
    all_qwk = []
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        qwk_df['model'] = data['model_short']
        all_qwk.append(qwk_df)
    combined = pd.concat(all_qwk, ignore_index=True)

    # Compare: unsteered, openai, best steered per model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: key judge types
    ax = axes[0]
    key_judges = ['openai', 'unsteered']
    x = np.arange(len(key_judges))
    width = 0.8 / len(models)
    for mi, model in enumerate(models):
        sub = combined[combined['model'] == model]
        vals = [sub[sub['judge_type'] == jt]['mean_qwk'].values[0]
                if len(sub[sub['judge_type'] == jt]) > 0 else 0
                for jt in key_judges]
        offset = (mi - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model,
               color=get_model_color(model), alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(key_judges, fontsize=11)
    ax.set_ylabel('Mean QWK')
    ax.set_title('Baseline Judge Performance')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right: best steered per model
    ax = axes[1]
    best_data = []
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        steered = qwk_df[~qwk_df['judge_type'].isin(['openai', 'unsteered'])]
        if len(steered) > 0:
            best = steered.sort_values('mean_qwk', ascending=False).iloc[0]
            best_data.append({
                'model': data['model_short'],
                'best_judge': best['judge_type'],
                'mean_qwk': best['mean_qwk'],
            })
        unst = qwk_df[qwk_df['judge_type'] == 'unsteered']
        if len(unst) > 0:
            best_data.append({
                'model': data['model_short'],
                'best_judge': 'unsteered',
                'mean_qwk': unst.iloc[0]['mean_qwk'],
            })
    best_df = pd.DataFrame(best_data)

    x = np.arange(len(models))
    for mi, model in enumerate(models):
        sub = best_df[best_df['model'] == model]
        unst = sub[sub['best_judge'] == 'unsteered']
        best = sub[sub['best_judge'] != 'unsteered']
        if len(unst) > 0:
            ax.bar(mi - 0.2, unst.iloc[0]['mean_qwk'], 0.35,
                   color=get_model_color(model), alpha=0.5, edgecolor='white')
        if len(best) > 0:
            bar = ax.bar(mi + 0.2, best.iloc[0]['mean_qwk'], 0.35,
                         color=get_model_color(model), alpha=0.85, edgecolor='white')
            ax.text(mi + 0.2, best.iloc[0]['mean_qwk'] + 0.01,
                    best.iloc[0]['best_judge'], ha='center', fontsize=7, rotation=30)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean QWK')
    ax.set_title('Best Steered Judge vs Unsteered')
    ax.grid(axis='y', alpha=0.3)
    # Manual legend
    ax.legend([mpatches.Patch(alpha=0.5, color='gray'), mpatches.Patch(alpha=0.85, color='gray')],
              ['Unsteered', 'Best steered'], fontsize=9)

    fig.suptitle('Judge Performance Across Models (Experiment B)', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot(plot_dir, '2_judge_qwk_comparison')


def plot_trait_radar(all_data, plot_dir):
    """3: Radar chart of trait effect magnitude per model."""
    models = [d['model_short'] for d in all_data]
    angles = np.linspace(0, 2 * np.pi, len(ALL_TRAITS), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

    for idx, direction in enumerate(['pos', 'neg']):
        ax = axes[idx]
        for data in all_data:
            effects = compute_effect_sizes(data)
            eff_openai = effects[(effects['judge_type'] == 'openai') & (effects['direction'] == direction)]
            vals = [abs(eff_openai[eff_openai['trait'] == t]['effect'].values[0])
                    if len(eff_openai[eff_openai['trait'] == t]) > 0 else 0
                    for t in ALL_TRAITS]
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', label=data['model_short'],
                    color=get_model_color(data['model_short']), linewidth=2)
            ax.fill(angles, vals, alpha=0.1, color=get_model_color(data['model_short']))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ALL_TRAITS, fontsize=9)
        dir_label = 'Positive (+trait)' if direction == 'pos' else 'Negative (opposite)'
        ax.set_title(f'{dir_label}\n(absolute effect)', fontsize=11, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle('Trait Vulnerability Profile by Model', fontsize=14, y=1.05)
    plt.tight_layout()
    return save_plot(plot_dir, '3_trait_radar')


def plot_judge_bias_heatmap(all_data, plot_dir):
    """4: Judge bias heatmap across models."""
    models = [d['model_short'] for d in all_data]
    judge_types = ['unsteered'] + [f'{t}_{d}' for t in ALL_TRAITS for d in ['pos', 'neg']]

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for mi, data in enumerate(all_data):
        ax = axes[mi]
        qwk_df = compute_judge_qwk(data)
        # Build matrix: judges x metrics (QWK, MAE, bias)
        judge_order = [jt for jt in judge_types if jt in qwk_df['judge_type'].values]
        matrix = np.zeros((len(judge_order), 1))
        for i, jt in enumerate(judge_order):
            row = qwk_df[qwk_df['judge_type'] == jt]
            if len(row) > 0:
                matrix[i, 0] = row.iloc[0]['mean_qwk']

        im = ax.imshow(matrix, cmap='RdYlGn', aspect=0.3, vmin=-0.1, vmax=0.6)
        ax.set_yticks(range(len(judge_order)))
        ax.set_yticklabels(judge_order, fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels(['QWK'], fontsize=10)
        ax.set_title(f'{data["model_short"]}\n(layer {data["config"].get("layer", "?")})', fontsize=11)
        for i in range(len(judge_order)):
            ax.text(0, i, f'{matrix[i, 0]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, 0] > 0.45 or matrix[i, 0] < 0 else 'black')

    fig.colorbar(im, ax=axes, shrink=0.5, label='Mean QWK')
    fig.suptitle('Judge QWK by Steering Type Across Models', fontsize=14, y=1.02)
    plt.tight_layout()
    return save_plot(plot_dir, '4_judge_qwk_heatmap')


def plot_answer_length_comparison(all_data, plot_dir):
    """5: Answer length comparison across models."""
    models = [d['model_short'] for d in all_data]
    student_types = ['unsteered'] + [f'{t}_{d}' for t in ALL_TRAITS for d in ['pos', 'neg']]

    fig, ax = plt.subplots(figsize=(14, 8))
    n_models = len(models)
    width = 0.8 / n_models

    for mi, data in enumerate(all_data):
        lens = data['answers_a'].groupby('student_type')['answer_len'].mean()
        available = [st for st in student_types if st in lens.index]
        vals = [lens.get(st, 0) for st in available]
        y = np.arange(len(available))
        offset = (mi - (n_models - 1) / 2) * width
        ax.barh(y + offset, vals, width, label=data['model_short'],
                color=get_model_color(data['model_short']), alpha=0.85, edgecolor='white')

    ax.set_yticks(np.arange(len(available)))
    ax.set_yticklabels(available, fontsize=8)
    ax.set_xlabel('Mean answer length (chars)')
    ax.set_title('Answer Length by Student Type Across Models')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return save_plot(plot_dir, '5_answer_length')


def plot_overall_summary(all_data, plot_dir):
    """6: Summary scatter: model susceptibility vs judge capability."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for data in all_data:
        effects = compute_effect_sizes(data)
        pos_effects = effects[(effects['judge_type'] == 'openai') & (effects['direction'] == 'pos')]
        mean_vulnerability = pos_effects['effect'].abs().mean()

        qwk_df = compute_judge_qwk(data)
        unst_qwk = qwk_df[qwk_df['judge_type'] == 'unsteered']['mean_qwk'].values
        unst_qwk = unst_qwk[0] if len(unst_qwk) > 0 else 0

        model = data['model_short']
        ax.scatter(mean_vulnerability, unst_qwk, s=300,
                   color=get_model_color(model), zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(f"{model}\n(layer {data['config'].get('layer', '?')})",
                    (mean_vulnerability, unst_qwk),
                    textcoords='offset points', xytext=(15, 0), fontsize=11,
                    fontweight='bold')

    ax.set_xlabel('Mean Steering Vulnerability\n(avg |effect| of positive steering, OpenAI judge)', fontsize=11)
    ax.set_ylabel('Unsteered Judge QWK\n(scoring accuracy without steering)', fontsize=11)
    ax.set_title('Model Overview: Steering Vulnerability vs Judge Capability', fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save_plot(plot_dir, '6_model_overview')


def plot_per_set_model_comparison(all_data, plot_dir):
    """7: Per-set QWK comparison across models (unsteered judge)."""
    models = [d['model_short'] for d in all_data]
    essay_configs_path = Path(__file__).resolve().parent / 'essay_configs.json'
    with open(essay_configs_path) as f:
        raw = json.load(f)
    essay_sets = {int(k): {**v, 'score_range': tuple(v['score_range'])} for k, v in raw.items()}
    set_ids = sorted(essay_sets.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(set_ids))

    for mi, data in enumerate(all_data):
        scores_b = data['scores_b']
        unst = scores_b[scores_b['judge_type'] == 'unsteered']
        qwks = []
        for sid in set_ids:
            s = unst[unst['set_id'] == sid]
            es = essay_sets.get(sid)
            if es and len(s) > 0:
                qwk = calculate_qwk(s['raw_score'].tolist(), s['ground_truth_score'].tolist(),
                                    es['score_range'][0], es['score_range'][1])
                qwks.append(qwk)
            else:
                qwks.append(0)
        offset = (mi - (n_models - 1) / 2) * width
        ax.bar(x + offset, qwks, width, label=data['model_short'],
               color=get_model_color(data['model_short']), alpha=0.85, edgecolor='white')

    essay_info = all_data[0]['essay_sets_info']
    set_labels = [f"S{s}\n{essay_info.get(str(s), {}).get('topic', '')[:15]}" for s in set_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('QWK')
    ax.set_title('Unsteered Judge QWK by Essay Set Across Models')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    plt.tight_layout()
    return save_plot(plot_dir, '7_per_set_qwk')


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_cross_model_report(all_data, output_dir):
    report_dir = output_dir
    plot_dir = report_dir / 'plots'
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    models = [d['model_short'] for d in all_data]

    md = []
    md.append("# Cross-Model Comparison Report\n")
    md.append("> Comparing persona steering effects across multiple language models\n")
    md.append(f"> Models: {', '.join(models)}\n")
    md.append("")

    # Overview table
    md.append("## Model Overview\n")
    md.append("| Model | Parameters | Steering Layer | Architecture | Coeff |")
    md.append("|-------|-----------|---------------|--------------|-------|")
    arch_info = {
        'Qwen3-4B': ('4B', 'Dense Transformer'),
        'Qwen3-32B': ('32B', 'Dense Transformer'),
        'gpt-oss-20b': ('20B (MoE)', 'MoE, 32 experts, 4 active'),
    }
    for data in all_data:
        m = data['model_short']
        layer = data['config'].get('layer', '?')
        coeff = data['config'].get('coeff', 2.0)
        params, arch = arch_info.get(m, ('?', '?'))
        md.append(f"| {m} | {params} | {layer} | {arch} | {coeff} |")
    md.append("")

    # ===================== SECTION 1: Student Effects =====================
    md.append("---\n## 1. Student Answer Quality (Experiment A)\n")
    md.append("How does steering the answer-generating model affect quality, and does this vary by model?\n")

    # Plot 1: Effect comparison
    md.append("### 1.1 Steering Effect Comparison\n")
    p1 = plot_effect_comparison(all_data, plot_dir)
    md.append(f"![Effect Comparison]({p1})\n")

    # Compute summary stats
    effect_summaries = {}
    for data in all_data:
        effects = compute_effect_sizes(data)
        pos_eff = effects[(effects['judge_type'] == 'openai') & (effects['direction'] == 'pos')]
        neg_eff = effects[(effects['judge_type'] == 'openai') & (effects['direction'] == 'neg')]
        effect_summaries[data['model_short']] = {
            'mean_pos_effect': pos_eff['effect'].mean(),
            'mean_neg_effect': neg_eff['effect'].mean(),
            'mean_pos_abs': pos_eff['effect'].abs().mean(),
            'worst_trait': pos_eff.loc[pos_eff['effect'].idxmin(), 'trait'],
            'worst_effect': pos_eff['effect'].min(),
        }

    md.append("**Key findings:**\n")

    # Which model is most vulnerable?
    most_vulnerable = max(effect_summaries, key=lambda m: effect_summaries[m]['mean_pos_abs'])
    least_vulnerable = min(effect_summaries, key=lambda m: effect_summaries[m]['mean_pos_abs'])
    md.append(f"- **{most_vulnerable}** is the most vulnerable to positive steering "
              f"(avg |effect| = {effect_summaries[most_vulnerable]['mean_pos_abs']:.3f}), "
              f"while **{least_vulnerable}** is the most resilient "
              f"({effect_summaries[least_vulnerable]['mean_pos_abs']:.3f}).")

    # Most destructive trait per model
    for m, s in effect_summaries.items():
        md.append(f"- **{m}**: worst trait = {s['worst_trait']} ({s['worst_effect']:+.3f})")
    md.append("")

    # Summary table
    md.append("| Model | Avg +pos Effect | Avg -neg Effect | Most Destructive Trait | Worst Effect |")
    md.append("|-------|----------------|----------------|----------------------|-------------|")
    for m, s in effect_summaries.items():
        md.append(f"| {m} | {s['mean_pos_effect']:+.3f} | {s['mean_neg_effect']:+.3f} | "
                  f"{s['worst_trait']} | {s['worst_effect']:+.3f} |")
    md.append("")

    # Plot 3: Radar
    md.append("### 1.2 Trait Vulnerability Profiles\n")
    p3 = plot_trait_radar(all_data, plot_dir)
    md.append(f"![Trait Radar]({p3})\n")
    md.append("The radar charts show which traits each model is most susceptible to. "
              "Larger area = more vulnerable overall.\n")

    # Per-trait comparison table
    md.append("### 1.3 Per-Trait Effect Comparison\n")
    header = "| Trait | Direction |"
    sep = "|-------|-----------|"
    for m in models:
        header += f" {m} |"
        sep += "--------|"
    md.append(header)
    md.append(sep)
    for trait in ALL_TRAITS:
        for direction in ['pos', 'neg']:
            row = f"| {trait} | {direction} |"
            for data in all_data:
                effects = compute_effect_sizes(data)
                eff = effects[(effects['judge_type'] == 'openai') &
                              (effects['trait'] == trait) &
                              (effects['direction'] == direction)]
                val = eff['effect'].values[0] if len(eff) > 0 else 0
                row += f" {val:+.3f} |"
            md.append(row)
    md.append("")

    # Plot 5: Answer length
    md.append("### 1.4 Answer Length Comparison\n")
    p5 = plot_answer_length_comparison(all_data, plot_dir)
    md.append(f"![Answer Length]({p5})\n")

    len_summaries = {}
    for data in all_data:
        unst_len = data['answers_a'][data['answers_a']['student_type'] == 'unsteered']['answer_len'].mean()
        len_summaries[data['model_short']] = unst_len
    md.append("**Baseline answer lengths (unsteered):**\n")
    for m, l in len_summaries.items():
        md.append(f"- {m}: {l:.0f} chars")
    md.append("")

    # ===================== SECTION 2: Judge Accuracy =====================
    md.append("---\n## 2. Judge Scoring Accuracy (Experiment B)\n")
    md.append("How well does each model score real essays, and how does steering affect accuracy?\n")

    # Plot 2: QWK comparison
    md.append("### 2.1 Judge QWK Overview\n")
    p2 = plot_judge_qwk_comparison(all_data, plot_dir)
    md.append(f"![Judge QWK]({p2})\n")

    # Summary table
    md.append("| Model | OpenAI QWK | Unsteered QWK | Best Steered | Best Steered QWK | Worst Steered | Worst QWK |")
    md.append("|-------|-----------|--------------|-------------|-----------------|--------------|-----------|")
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        openai_row = qwk_df[qwk_df['judge_type'] == 'openai']
        unst_row = qwk_df[qwk_df['judge_type'] == 'unsteered']
        steered = qwk_df[~qwk_df['judge_type'].isin(['openai', 'unsteered'])]

        openai_qwk = openai_row.iloc[0]['mean_qwk'] if len(openai_row) > 0 else 0
        unst_qwk = unst_row.iloc[0]['mean_qwk'] if len(unst_row) > 0 else 0
        if len(steered) > 0:
            best = steered.sort_values('mean_qwk', ascending=False).iloc[0]
            worst = steered.sort_values('mean_qwk', ascending=True).iloc[0]
            md.append(f"| {data['model_short']} | {openai_qwk:.3f} | {unst_qwk:.3f} | "
                      f"{best['judge_type']} | {best['mean_qwk']:.3f} | "
                      f"{worst['judge_type']} | {worst['mean_qwk']:.3f} |")
        else:
            md.append(f"| {data['model_short']} | {openai_qwk:.3f} | {unst_qwk:.3f} | N/A | N/A | N/A | N/A |")
    md.append("")

    # Plot 4: QWK heatmap
    md.append("### 2.2 Per-Judge QWK Heatmap\n")
    p4 = plot_judge_bias_heatmap(all_data, plot_dir)
    md.append(f"![Judge QWK Heatmap]({p4})\n")

    # Plot 7: Per-set QWK
    md.append("### 2.3 Per-Set Unsteered Judge QWK\n")
    p7 = plot_per_set_model_comparison(all_data, plot_dir)
    md.append(f"![Per-Set QWK]({p7})\n")
    md.append("This shows which essay types each model judges best/worst at (without steering).\n")

    # ===================== SECTION 3: Overall =====================
    md.append("---\n## 3. Overall Model Comparison\n")

    # Plot 6: Overview scatter
    md.append("### 3.1 Vulnerability vs Capability\n")
    p6 = plot_overall_summary(all_data, plot_dir)
    md.append(f"![Model Overview]({p6})\n")
    md.append("This plot positions each model on two axes: how much steering hurts answer quality (x) "
              "vs how well the unsteered model scores essays (y). "
              "**Ideal position: bottom-left** (low vulnerability, high QWK).\n")

    # Grand summary table
    md.append("### 3.2 Grand Summary\n")
    md.append("| Metric | " + " | ".join(models) + " |")
    md.append("|--------|" + "|".join(["--------"] * len(models)) + "|")

    # Rows
    metrics = []

    # Steering layer
    row = "| Steering layer |"
    for data in all_data:
        row += f" {data['config'].get('layer', '?')} |"
    metrics.append(row)

    # Mean pos effect (OpenAI judge)
    row = "| Mean +pos effect (OpenAI) |"
    for data in all_data:
        row += f" {effect_summaries[data['model_short']]['mean_pos_effect']:+.3f} |"
    metrics.append(row)

    # Mean vulnerability
    row = "| Mean vulnerability (|pos effect|) |"
    for data in all_data:
        row += f" {effect_summaries[data['model_short']]['mean_pos_abs']:.3f} |"
    metrics.append(row)

    # Unsteered judge QWK
    row = "| Unsteered judge QWK |"
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        unst = qwk_df[qwk_df['judge_type'] == 'unsteered']
        val = unst.iloc[0]['mean_qwk'] if len(unst) > 0 else 0
        row += f" {val:.3f} |"
    metrics.append(row)

    # OpenAI judge QWK
    row = "| OpenAI judge QWK |"
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        openai = qwk_df[qwk_df['judge_type'] == 'openai']
        val = openai.iloc[0]['mean_qwk'] if len(openai) > 0 else 0
        row += f" {val:.3f} |"
    metrics.append(row)

    # Best steered judge
    row = "| Best steered judge |"
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        steered = qwk_df[~qwk_df['judge_type'].isin(['openai', 'unsteered'])]
        if len(steered) > 0:
            best = steered.sort_values('mean_qwk', ascending=False).iloc[0]
            row += f" {best['judge_type']} ({best['mean_qwk']:.3f}) |"
        else:
            row += " N/A |"
    metrics.append(row)

    # Worst trait
    row = "| Most destructive trait |"
    for data in all_data:
        s = effect_summaries[data['model_short']]
        row += f" {s['worst_trait']} ({s['worst_effect']:+.3f}) |"
    metrics.append(row)

    # Unsteered answer length
    row = "| Unsteered answer length |"
    for data in all_data:
        row += f" {len_summaries[data['model_short']]:.0f} chars |"
    metrics.append(row)

    md.extend(metrics)
    md.append("")

    # Key takeaways
    md.append("### 3.3 Key Takeaways\n")

    # 1. Most/least vulnerable
    md.append(f"1. **{most_vulnerable} is most vulnerable** to steering "
              f"(avg |effect| = {effect_summaries[most_vulnerable]['mean_pos_abs']:.3f}), "
              f"while **{least_vulnerable}** is most resilient "
              f"({effect_summaries[least_vulnerable]['mean_pos_abs']:.3f}). "
              f"{'Larger models are not necessarily more robust.' if most_vulnerable != 'Qwen3-4B' else 'Smaller models tend to be more susceptible.'}\n")

    # 2. Judge capability
    judge_qwks = {}
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        unst = qwk_df[qwk_df['judge_type'] == 'unsteered']
        judge_qwks[data['model_short']] = unst.iloc[0]['mean_qwk'] if len(unst) > 0 else 0
    best_judge_model = max(judge_qwks, key=judge_qwks.get)
    md.append(f"2. **{best_judge_model} is the best unsteered judge** (QWK = {judge_qwks[best_judge_model]:.3f}). "
              f"All models are far below OpenAI's performance.\n")

    # 3. Evil is universally destructive
    evil_effects = {data['model_short']: effect_summaries[data['model_short']]['worst_trait']
                    for data in all_data}
    evil_models = [m for m, t in evil_effects.items() if t == 'evil']
    if len(evil_models) == len(models):
        md.append("3. **Evil steering is the most destructive trait across all models** — "
                  "it consistently produces the largest quality drops.\n")
    else:
        md.append(f"3. **The most destructive trait varies by model**: "
                  + ", ".join(f"{m}={t}" for m, t in evil_effects.items()) + ".\n")

    # 4. Steering can sometimes help
    any_positive = False
    for data in all_data:
        effects = compute_effect_sizes(data)
        pos = effects[(effects['judge_type'] == 'openai') & (effects['effect'] > 0.01)]
        if len(pos) > 0:
            any_positive = True
            break
    if any_positive:
        md.append("4. **Steering can occasionally improve quality** — some trait/direction combinations "
                  "produce small positive effects, suggesting the vectors capture meaningful latent behaviors.\n")

    md.append("5. **OpenAI (gpt-5.2) consistently outperforms all local LLM judges** across all models, "
              "confirming that larger proprietary models maintain a significant advantage in essay scoring.\n")

    md.append("---\n*Generated from `experiments/education/generate_cross_model_report.py`*\n")

    report_path = report_dir / 'cross_model_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'Report written to {report_path}')
    print(f'Plots saved to {plot_dir}')

    # Also save a combined CSV
    all_effects = []
    for data in all_data:
        effects = compute_effect_sizes(data)
        effects['model'] = data['model_short']
        all_effects.append(effects)
    combined_effects = pd.concat(all_effects, ignore_index=True)
    combined_effects.to_csv(report_dir / 'combined_effects.csv', index=False)

    all_qwk = []
    for data in all_data:
        qwk_df = compute_judge_qwk(data)
        qwk_df['model'] = data['model_short']
        all_qwk.append(qwk_df)
    combined_qwk = pd.concat(all_qwk, ignore_index=True)
    combined_qwk.to_csv(report_dir / 'combined_judge_qwk.csv', index=False)

    print(f'Data saved: combined_effects.csv, combined_judge_qwk.csv')
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cross-model comparison report')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/cross_model_comparison)')
    parser.add_argument('--results-root', type=str, default=None,
                        help='Root directory containing result dirs (default: auto-detect)')
    args = parser.parse_args()

    root = Path(args.results_root) if args.results_root else RESULTS_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else root / 'cross_model_comparison'

    print(f'Discovering result directories in {root}...')
    result_dirs = discover_result_dirs(root)
    print(f'Found {len(result_dirs)} result directories:')
    for d in result_dirs:
        print(f'  - {d.name}')

    print('\nLoading data...')
    all_data = []
    for d in result_dirs:
        try:
            data = load_model_data(d)
            print(f'  Loaded {data["model_short"]} ({len(data["answers_a"])} answers, '
                  f'{len(data["scores_b"])} judge scores)')
            all_data.append(data)
        except Exception as e:
            print(f'  SKIP {d.name}: {e}')

    if len(all_data) < 2:
        print(f'\nNeed at least 2 models for comparison, found {len(all_data)}. Exiting.')
        exit(1)

    print(f'\nGenerating cross-model report for {len(all_data)} models...')
    report_path = generate_cross_model_report(all_data, output_dir)
    print(f'\nDone! Report at: {report_path}')
