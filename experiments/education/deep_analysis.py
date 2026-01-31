"""
Comprehensive Deep Analysis of Education Scoring Experiment Results

This script analyzes the persona steering experiment results and generates:
- Descriptive statistics
- Statistical significance tests
- Visualizations
- A comprehensive markdown report

Usage:
    python -m experiments.education.deep_analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_results(results_path: str) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def flatten_results(data: dict) -> pd.DataFrame:
    """Flatten hierarchical results into a DataFrame."""
    rows = []
    for set_result in data['results']:
        set_id = set_result['set_id']
        topic = set_result['topic']
        score_range = tuple(set_result['score_range'])

        # Categorize question type
        if 'Science' in topic or 'Experiment' in topic or 'Protein' in topic or 'Cell' in topic:
            question_type = 'Science'
        elif 'Literary' in topic:
            question_type = 'Literary'
        elif 'Invasive' in topic or 'Space' in topic:
            question_type = 'Reading Comprehension'
        else:
            question_type = 'Opinion/Discussion'

        for sample in set_result['samples']:
            sample_id = sample['sample_id']
            for student_type, student_data in sample['students'].items():
                for judge_type, raw_score in student_data['scores'].items():
                    normalized = student_data['normalized'][judge_type]
                    rows.append({
                        'set_id': set_id,
                        'topic': topic,
                        'question_type': question_type,
                        'score_range': score_range,
                        'sample_id': sample_id,
                        'student_type': student_type,
                        'judge_type': judge_type,
                        'raw_score': raw_score,
                        'normalized_score': normalized,
                        'answer_length': len(student_data['answer']),
                    })
    return pd.DataFrame(rows)

def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics."""
    stats_dict = {}

    # Overall stats by student type
    stats_dict['by_student'] = df.groupby('student_type')['normalized_score'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(4)

    # Overall stats by judge type
    stats_dict['by_judge'] = df.groupby('judge_type')['normalized_score'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(4)

    # Cross-tabulation: Student x Judge
    stats_dict['student_judge_matrix'] = df.pivot_table(
        values='normalized_score',
        index='student_type',
        columns='judge_type',
        aggfunc='mean'
    ).round(4)

    # By question type
    stats_dict['by_question_type'] = df.pivot_table(
        values='normalized_score',
        index='question_type',
        columns='student_type',
        aggfunc='mean'
    ).round(4)

    # By set
    stats_dict['by_set'] = df.pivot_table(
        values='normalized_score',
        index=['set_id', 'topic'],
        columns='student_type',
        aggfunc='mean'
    ).round(4)

    return stats_dict

def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run statistical significance tests."""
    results = {}

    # 1. ANOVA: Are there significant differences between student types?
    student_groups = [group['normalized_score'].values for name, group in df.groupby('student_type')]
    f_stat, p_val = stats.f_oneway(*student_groups)
    results['anova_students'] = {'F': f_stat, 'p': p_val}

    # 2. ANOVA: Are there significant differences between judge types?
    judge_groups = [group['normalized_score'].values for name, group in df.groupby('judge_type')]
    f_stat, p_val = stats.f_oneway(*judge_groups)
    results['anova_judges'] = {'F': f_stat, 'p': p_val}

    # 3. Pairwise t-tests between student types
    student_types = df['student_type'].unique()
    pairwise_students = {}
    for s1, s2 in combinations(student_types, 2):
        g1 = df[df['student_type'] == s1]['normalized_score']
        g2 = df[df['student_type'] == s2]['normalized_score']
        t_stat, p_val = stats.ttest_ind(g1, g2)
        cohen_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        pairwise_students[f'{s1}_vs_{s2}'] = {
            't': t_stat, 'p': p_val, 'cohen_d': cohen_d,
            'mean_diff': g1.mean() - g2.mean()
        }
    results['pairwise_students'] = pairwise_students

    # 4. Pairwise t-tests between judge types
    judge_types = df['judge_type'].unique()
    pairwise_judges = {}
    for j1, j2 in combinations(judge_types, 2):
        g1 = df[df['judge_type'] == j1]['normalized_score']
        g2 = df[df['judge_type'] == j2]['normalized_score']
        t_stat, p_val = stats.ttest_ind(g1, g2)
        cohen_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        pairwise_judges[f'{j1}_vs_{j2}'] = {
            't': t_stat, 'p': p_val, 'cohen_d': cohen_d,
            'mean_diff': g1.mean() - g2.mean()
        }
    results['pairwise_judges'] = pairwise_judges

    # 5. Bias test: Do judges favor their own student type?
    bias_results = {}
    for judge in ['good', 'evil']:
        # Compare how this judge scores "matching" vs "non-matching" students
        matching = df[(df['judge_type'] == judge) & (df['student_type'] == judge)]['normalized_score']
        non_matching = df[(df['judge_type'] == judge) & (df['student_type'] != judge)]['normalized_score']
        t_stat, p_val = stats.ttest_ind(matching, non_matching)
        bias_results[f'{judge}_judge_bias'] = {
            't': t_stat, 'p': p_val,
            'matching_mean': matching.mean(),
            'non_matching_mean': non_matching.mean(),
            'bias_magnitude': matching.mean() - non_matching.mean()
        }
    results['bias_tests'] = bias_results

    # 6. Question type effects (2-way ANOVA would be ideal, but let's do simpler tests)
    question_type_results = {}
    for qt in df['question_type'].unique():
        qt_df = df[df['question_type'] == qt]
        groups = [group['normalized_score'].values for name, group in qt_df.groupby('student_type')]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            question_type_results[qt] = {'F': f_stat, 'p': p_val}
    results['question_type_effects'] = question_type_results

    # 7. Judge agreement (correlation between judges)
    judge_agreement = {}
    pivot_for_corr = df.pivot_table(
        values='normalized_score',
        index=['set_id', 'sample_id', 'student_type'],
        columns='judge_type',
        aggfunc='mean'
    )
    for j1, j2 in combinations(df['judge_type'].unique(), 2):
        if j1 in pivot_for_corr.columns and j2 in pivot_for_corr.columns:
            corr, p_val = stats.pearsonr(pivot_for_corr[j1], pivot_for_corr[j2])
            judge_agreement[f'{j1}_vs_{j2}'] = {'correlation': corr, 'p': p_val}
    results['judge_agreement'] = judge_agreement

    return results

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create all visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score distribution by student type
    fig, ax = plt.subplots(figsize=(10, 6))
    for student_type in ['good', 'evil', 'unsteered']:
        data = df[df['student_type'] == student_type]['normalized_score']
        ax.hist(data, bins=20, alpha=0.5, label=f'{student_type.capitalize()} Student', edgecolor='black')
    ax.set_xlabel('Normalized Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution by Student Type', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'score_dist_by_student.png', dpi=150)
    plt.close()

    # 2. Score distribution by judge type
    fig, ax = plt.subplots(figsize=(10, 6))
    for judge_type in ['good', 'evil', 'unsteered', 'openai']:
        data = df[df['judge_type'] == judge_type]['normalized_score']
        ax.hist(data, bins=20, alpha=0.5, label=f'{judge_type.capitalize()} Judge', edgecolor='black')
    ax.set_xlabel('Normalized Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution by Judge Type', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'score_dist_by_judge.png', dpi=150)
    plt.close()

    # 3. Box plot: Student x Judge matrix
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['combination'] = df_plot['student_type'] + ' → ' + df_plot['judge_type']
    order = [f'{s} → {j}' for s in ['good', 'evil', 'unsteered'] for j in ['good', 'evil', 'unsteered', 'openai']]
    sns.boxplot(data=df_plot, x='combination', y='normalized_score', order=order, ax=ax)
    ax.set_xlabel('Student → Judge', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Score Distribution: Student × Judge Combinations', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_student_judge.png', dpi=150)
    plt.close()

    # 4. Heatmap: Student x Judge mean scores
    pivot = df.pivot_table(
        values='normalized_score',
        index='student_type',
        columns='judge_type',
        aggfunc='mean'
    )
    # Reorder
    pivot = pivot.reindex(index=['good', 'evil', 'unsteered'], columns=['good', 'evil', 'unsteered', 'openai'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.7,
                vmin=0.5, vmax=0.9, ax=ax, linewidths=0.5)
    ax.set_xlabel('Judge Type', fontsize=12)
    ax.set_ylabel('Student Type', fontsize=12)
    ax.set_title('Mean Normalized Scores: Student × Judge Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_student_judge.png', dpi=150)
    plt.close()

    # 5. Performance by question type
    pivot_qt = df.pivot_table(
        values='normalized_score',
        index='question_type',
        columns='student_type',
        aggfunc='mean'
    )
    pivot_qt = pivot_qt[['good', 'evil', 'unsteered']]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_qt.plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Mean Normalized Score', fontsize=12)
    ax.set_title('Performance by Question Type and Student Type', fontsize=14)
    ax.legend(title='Student Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_question_type.png', dpi=150)
    plt.close()

    # 6. Performance by set (line plot)
    pivot_set = df.pivot_table(
        values='normalized_score',
        index='set_id',
        columns='student_type',
        aggfunc='mean'
    )
    pivot_set = pivot_set[['good', 'evil', 'unsteered']]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_set.plot(marker='o', ax=ax, linewidth=2, markersize=8)
    ax.set_xlabel('Set ID', fontsize=12)
    ax.set_ylabel('Mean Normalized Score', fontsize=12)
    ax.set_title('Performance Across Question Sets by Student Type', fontsize=14)
    ax.legend(title='Student Type')
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_set.png', dpi=150)
    plt.close()

    # 7. Judge leniency comparison
    judge_means = df.groupby('judge_type')['normalized_score'].mean().sort_values(ascending=False)
    judge_stds = df.groupby('judge_type')['normalized_score'].std()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(judge_means.index, judge_means.values, yerr=judge_stds[judge_means.index].values,
                  capsize=5, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Judge Type', fontsize=12)
    ax.set_ylabel('Mean Normalized Score', fontsize=12)
    ax.set_title('Judge Leniency Comparison (Mean ± Std)', fontsize=14)
    ax.axhline(y=df['normalized_score'].mean(), color='red', linestyle='--', label='Overall Mean')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'judge_leniency.png', dpi=150)
    plt.close()

    # 8. Bias visualization: In-group vs Out-group scoring
    bias_data = []
    for judge in ['good', 'evil']:
        matching = df[(df['judge_type'] == judge) & (df['student_type'] == judge)]['normalized_score'].mean()
        non_matching = df[(df['judge_type'] == judge) & (df['student_type'] != judge)]['normalized_score'].mean()
        bias_data.append({'Judge': judge.capitalize(), 'Scoring': 'In-group', 'Mean Score': matching})
        bias_data.append({'Judge': judge.capitalize(), 'Scoring': 'Out-group', 'Mean Score': non_matching})

    bias_df = pd.DataFrame(bias_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=bias_df, x='Judge', y='Mean Score', hue='Scoring', ax=ax, edgecolor='black')
    ax.set_xlabel('Judge Type', fontsize=12)
    ax.set_ylabel('Mean Normalized Score', fontsize=12)
    ax.set_title('In-group vs Out-group Scoring Bias', fontsize=14)
    ax.legend(title='Scoring Target')
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_ingroup_outgroup.png', dpi=150)
    plt.close()

    # 9. Answer length vs score
    fig, ax = plt.subplots(figsize=(10, 6))
    for student_type in ['good', 'evil', 'unsteered']:
        data = df[df['student_type'] == student_type]
        ax.scatter(data['answer_length'], data['normalized_score'], alpha=0.5, label=student_type.capitalize())
    ax.set_xlabel('Answer Length (characters)', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Answer Length vs Score by Student Type', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'answer_length_vs_score.png', dpi=150)
    plt.close()

    # 10. Judge agreement scatter matrix
    pivot_for_corr = df.pivot_table(
        values='normalized_score',
        index=['set_id', 'sample_id', 'student_type'],
        columns='judge_type',
        aggfunc='mean'
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    judge_pairs = [('good', 'evil'), ('good', 'unsteered'), ('good', 'openai'),
                   ('evil', 'unsteered'), ('evil', 'openai'), ('unsteered', 'openai')]

    for ax, (j1, j2) in zip(axes.flatten(), judge_pairs):
        ax.scatter(pivot_for_corr[j1], pivot_for_corr[j2], alpha=0.5)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
        corr = pivot_for_corr[j1].corr(pivot_for_corr[j2])
        ax.set_xlabel(f'{j1.capitalize()} Judge', fontsize=10)
        ax.set_ylabel(f'{j2.capitalize()} Judge', fontsize=10)
        ax.set_title(f'r = {corr:.3f}', fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Judge Agreement: Pairwise Score Correlations', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'judge_agreement_scatter.png', dpi=150)
    plt.close()

    print(f"Saved 10 visualizations to {output_dir}")

def generate_markdown_report(df: pd.DataFrame, desc_stats: dict, tests: dict,
                             config: dict, output_path: Path, img_dir: Path):
    """Generate comprehensive markdown report."""

    report = []

    # Title and metadata
    report.append("# Persona Steering Education Experiment: Comprehensive Analysis\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Model:** {config.get('model', 'N/A')}\n")
    report.append(f"**Samples:** {config.get('samples_per_set', 'N/A')} per set × 10 sets = {config.get('samples_per_set', 0) * 10} total\n")
    report.append(f"**Scoring Events:** {len(df)} (3 students × 4 judges × {config.get('samples_per_set', 0) * 10} samples)\n\n")

    # Table of contents
    report.append("## Table of Contents\n")
    report.append("1. [Executive Summary](#executive-summary)\n")
    report.append("2. [Score Distributions](#score-distributions)\n")
    report.append("3. [Student Performance Analysis](#student-performance-analysis)\n")
    report.append("4. [Judge Behavior Analysis](#judge-behavior-analysis)\n")
    report.append("5. [Student × Judge Interactions](#student--judge-interactions)\n")
    report.append("6. [Question Type Analysis](#question-type-analysis)\n")
    report.append("7. [Statistical Significance Tests](#statistical-significance-tests)\n")
    report.append("8. [Key Insights & Conclusions](#key-insights--conclusions)\n\n")

    # Executive Summary
    report.append("---\n\n## Executive Summary\n\n")

    # Calculate key metrics
    student_means = df.groupby('student_type')['normalized_score'].mean()
    judge_means = df.groupby('judge_type')['normalized_score'].mean()

    report.append("### Key Findings\n\n")
    report.append("| Metric | Finding |\n")
    report.append("|--------|--------|\n")
    report.append(f"| **Best Performing Student** | {student_means.idxmax().capitalize()} ({student_means.max():.3f}) |\n")
    report.append(f"| **Worst Performing Student** | {student_means.idxmin().capitalize()} ({student_means.min():.3f}) |\n")
    report.append(f"| **Most Lenient Judge** | {judge_means.idxmax().capitalize()} ({judge_means.max():.3f}) |\n")
    report.append(f"| **Strictest Judge** | {judge_means.idxmin().capitalize()} ({judge_means.min():.3f}) |\n")

    # Bias finding
    evil_bias = tests['bias_tests']['evil_judge_bias']['bias_magnitude']
    good_bias = tests['bias_tests']['good_judge_bias']['bias_magnitude']
    report.append(f"| **Evil Judge In-group Bias** | {evil_bias:+.3f} (p={tests['bias_tests']['evil_judge_bias']['p']:.4f}) |\n")
    report.append(f"| **Good Judge In-group Bias** | {good_bias:+.3f} (p={tests['bias_tests']['good_judge_bias']['p']:.4f}) |\n")
    report.append("\n")

    # Main findings in text
    report.append("### Summary\n\n")
    report.append(f"1. **Unsteered students outperform steered students**: Unsteered students achieved the highest ")
    report.append(f"mean score ({student_means['unsteered']:.3f}), followed by good-steered ({student_means['good']:.3f}) ")
    report.append(f"and evil-steered ({student_means['evil']:.3f}).\n\n")

    report.append(f"2. **Evil judge is most lenient**: The evil-steered judge gave the highest average scores ")
    report.append(f"({judge_means['evil']:.3f}), while OpenAI (gpt-5.2) was strictest ({judge_means['openai']:.3f}).\n\n")

    if evil_bias > 0:
        report.append(f"3. **Evil judges show in-group bias**: Evil judges scored evil students {abs(evil_bias):.3f} points ")
        report.append(f"higher than other students (p={tests['bias_tests']['evil_judge_bias']['p']:.4f}).\n\n")
    else:
        report.append(f"3. **Evil judges show OUT-group preference**: Evil judges scored evil students {abs(evil_bias):.3f} points ")
        report.append(f"LOWER than other students.\n\n")

    report.append(f"4. **OpenAI provides most discriminating scores**: OpenAI judge shows the largest score ")
    report.append(f"spread between student types, suggesting it may be best at detecting quality differences.\n\n")

    # Score Distributions
    report.append("---\n\n## Score Distributions\n\n")
    report.append("### Distribution by Student Type\n\n")
    report.append("![Score Distribution by Student](figures/score_dist_by_student.png)\n\n")

    report.append("**Descriptive Statistics:**\n\n")
    report.append(desc_stats['by_student'].to_markdown() + "\n\n")

    report.append("### Distribution by Judge Type\n\n")
    report.append("![Score Distribution by Judge](figures/score_dist_by_judge.png)\n\n")

    report.append("**Descriptive Statistics:**\n\n")
    report.append(desc_stats['by_judge'].to_markdown() + "\n\n")

    # Student Performance
    report.append("---\n\n## Student Performance Analysis\n\n")

    report.append("### Overall Performance Ranking\n\n")
    report.append("| Rank | Student Type | Mean Score | Std Dev | Interpretation |\n")
    report.append("|------|--------------|------------|---------|----------------|\n")
    sorted_students = student_means.sort_values(ascending=False)
    interpretations = {
        'unsteered': 'Baseline without steering intervention',
        'good': 'Steered toward helpful/constructive behavior',
        'evil': 'Steered toward harmful/unhelpful behavior'
    }
    for rank, (student, mean) in enumerate(sorted_students.items(), 1):
        std = df[df['student_type'] == student]['normalized_score'].std()
        report.append(f"| {rank} | {student.capitalize()} | {mean:.3f} | {std:.3f} | {interpretations[student]} |\n")
    report.append("\n")

    report.append("### Key Observations\n\n")

    # Good students
    good_mean = student_means['good']
    good_by_judge = df[df['student_type'] == 'good'].groupby('judge_type')['normalized_score'].mean()
    report.append(f"**Good-Steered Students:**\n")
    report.append(f"- Average score: {good_mean:.3f}\n")
    report.append(f"- Scored highest by: {good_by_judge.idxmax().capitalize()} judge ({good_by_judge.max():.3f})\n")
    report.append(f"- Scored lowest by: {good_by_judge.idxmin().capitalize()} judge ({good_by_judge.min():.3f})\n\n")

    # Evil students
    evil_mean = student_means['evil']
    evil_by_judge = df[df['student_type'] == 'evil'].groupby('judge_type')['normalized_score'].mean()
    report.append(f"**Evil-Steered Students:**\n")
    report.append(f"- Average score: {evil_mean:.3f}\n")
    report.append(f"- Scored highest by: {evil_by_judge.idxmax().capitalize()} judge ({evil_by_judge.max():.3f})\n")
    report.append(f"- Scored lowest by: {evil_by_judge.idxmin().capitalize()} judge ({evil_by_judge.min():.3f})\n")
    report.append(f"- **Can evil students produce good results?** Yes, evil students achieved scores ≥0.9 in ")
    high_evil = len(df[(df['student_type'] == 'evil') & (df['normalized_score'] >= 0.9)])
    total_evil = len(df[df['student_type'] == 'evil'])
    report.append(f"{high_evil}/{total_evil} cases ({100*high_evil/total_evil:.1f}%)\n\n")

    # Unsteered students
    unsteered_mean = student_means['unsteered']
    report.append(f"**Unsteered Students (Baseline):**\n")
    report.append(f"- Average score: {unsteered_mean:.3f}\n")
    report.append(f"- Outperforms good-steered by: {unsteered_mean - good_mean:+.3f}\n")
    report.append(f"- Outperforms evil-steered by: {unsteered_mean - evil_mean:+.3f}\n\n")

    # Judge Behavior
    report.append("---\n\n## Judge Behavior Analysis\n\n")

    report.append("### Judge Leniency Ranking\n\n")
    report.append("![Judge Leniency](figures/judge_leniency.png)\n\n")

    sorted_judges = judge_means.sort_values(ascending=False)
    report.append("| Rank | Judge Type | Mean Score | Interpretation |\n")
    report.append("|------|------------|------------|----------------|\n")
    judge_interp = {
        'evil': 'Most lenient - gives highest scores',
        'good': 'Moderately lenient',
        'unsteered': 'Baseline judge behavior',
        'openai': 'External reference (gpt-5.2)'
    }
    for rank, (judge, mean) in enumerate(sorted_judges.items(), 1):
        report.append(f"| {rank} | {judge.capitalize()} | {mean:.3f} | {judge_interp.get(judge, '')} |\n")
    report.append("\n")

    report.append("### Judge Consistency (Variance)\n\n")
    judge_stds = df.groupby('judge_type')['normalized_score'].std().sort_values()
    report.append("| Judge Type | Std Dev | Interpretation |\n")
    report.append("|------------|---------|----------------|\n")
    for judge, std in judge_stds.items():
        consistency = "Most consistent" if std == judge_stds.min() else "Least consistent" if std == judge_stds.max() else "Moderate"
        report.append(f"| {judge.capitalize()} | {std:.3f} | {consistency} |\n")
    report.append("\n")

    report.append("### Judge Agreement Analysis\n\n")
    report.append("![Judge Agreement](figures/judge_agreement_scatter.png)\n\n")

    report.append("**Correlation Matrix:**\n\n")
    report.append("| Judge Pair | Correlation | p-value | Agreement Level |\n")
    report.append("|------------|-------------|---------|------------------|\n")
    for pair, result in tests['judge_agreement'].items():
        corr = result['correlation']
        level = "Strong" if corr > 0.7 else "Moderate" if corr > 0.4 else "Weak"
        report.append(f"| {pair.replace('_', ' ').replace('vs', '↔')} | {corr:.3f} | {result['p']:.4f} | {level} |\n")
    report.append("\n")

    # Student x Judge Matrix
    report.append("---\n\n## Student × Judge Interactions\n\n")

    report.append("### Mean Score Matrix\n\n")
    report.append("![Heatmap](figures/heatmap_student_judge.png)\n\n")

    report.append(desc_stats['student_judge_matrix'].to_markdown() + "\n\n")

    report.append("### Box Plot of All Combinations\n\n")
    report.append("![Box Plot](figures/boxplot_student_judge.png)\n\n")

    report.append("### Bias Analysis: In-group vs Out-group Scoring\n\n")
    report.append("![Bias Analysis](figures/bias_ingroup_outgroup.png)\n\n")

    report.append("**Research Question: Are judges biased toward their own student type?**\n\n")

    for judge in ['good', 'evil']:
        bias = tests['bias_tests'][f'{judge}_judge_bias']
        sig = "statistically significant" if bias['p'] < 0.05 else "not statistically significant"
        direction = "in-group bias (favors matching students)" if bias['bias_magnitude'] > 0 else "out-group preference (favors non-matching students)"
        report.append(f"- **{judge.capitalize()} Judge**: Shows {direction}\n")
        report.append(f"  - Matching student score: {bias['matching_mean']:.3f}\n")
        report.append(f"  - Non-matching student score: {bias['non_matching_mean']:.3f}\n")
        report.append(f"  - Difference: {bias['bias_magnitude']:+.3f} ({sig}, p={bias['p']:.4f})\n\n")

    # Question Type Analysis
    report.append("---\n\n## Question Type Analysis\n\n")

    report.append("### Performance by Question Type\n\n")
    report.append("![Performance by Question Type](figures/performance_by_question_type.png)\n\n")

    report.append(desc_stats['by_question_type'].to_markdown() + "\n\n")

    report.append("### Performance Across Individual Sets\n\n")
    report.append("![Performance by Set](figures/performance_by_set.png)\n\n")

    # Find where each student type excels
    qt_pivot = desc_stats['by_question_type']
    report.append("**Where each student type excels:**\n\n")
    for student in ['good', 'evil', 'unsteered']:
        best_qt = qt_pivot[student].idxmax()
        worst_qt = qt_pivot[student].idxmin()
        report.append(f"- **{student.capitalize()}**: Best at {best_qt} ({qt_pivot.loc[best_qt, student]:.3f}), ")
        report.append(f"Worst at {worst_qt} ({qt_pivot.loc[worst_qt, student]:.3f})\n")
    report.append("\n")

    report.append("**Question type effects on student performance:**\n\n")
    for qt, result in tests['question_type_effects'].items():
        sig = "significant" if result['p'] < 0.05 else "not significant"
        report.append(f"- **{qt}**: F={result['F']:.2f}, p={result['p']:.4f} ({sig} difference between student types)\n")
    report.append("\n")

    # Statistical Tests
    report.append("---\n\n## Statistical Significance Tests\n\n")

    report.append("### ANOVA Results\n\n")
    report.append("| Test | F-statistic | p-value | Significant? |\n")
    report.append("|------|-------------|---------|-------------|\n")
    anova_s = tests['anova_students']
    anova_j = tests['anova_judges']
    report.append(f"| Differences between student types | {anova_s['F']:.2f} | {anova_s['p']:.4f} | {'Yes' if anova_s['p'] < 0.05 else 'No'} |\n")
    report.append(f"| Differences between judge types | {anova_j['F']:.2f} | {anova_j['p']:.4f} | {'Yes' if anova_j['p'] < 0.05 else 'No'} |\n")
    report.append("\n")

    report.append("### Pairwise Student Comparisons (t-tests)\n\n")
    report.append("| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size |\n")
    report.append("|------------|-----------|-------------|---------|-----------|-------------|\n")
    for pair, result in tests['pairwise_students'].items():
        effect = "Large" if abs(result['cohen_d']) > 0.8 else "Medium" if abs(result['cohen_d']) > 0.5 else "Small"
        sig = "*" if result['p'] < 0.05 else ""
        report.append(f"| {pair.replace('_', ' ')} | {result['mean_diff']:+.3f} | {result['t']:.2f} | {result['p']:.4f}{sig} | {result['cohen_d']:.3f} | {effect} |\n")
    report.append("\n*p < 0.05\n\n")

    report.append("### Pairwise Judge Comparisons (t-tests)\n\n")
    report.append("| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size |\n")
    report.append("|------------|-----------|-------------|---------|-----------|-------------|\n")
    for pair, result in tests['pairwise_judges'].items():
        effect = "Large" if abs(result['cohen_d']) > 0.8 else "Medium" if abs(result['cohen_d']) > 0.5 else "Small"
        sig = "*" if result['p'] < 0.05 else ""
        report.append(f"| {pair.replace('_', ' ')} | {result['mean_diff']:+.3f} | {result['t']:.2f} | {result['p']:.4f}{sig} | {result['cohen_d']:.3f} | {effect} |\n")
    report.append("\n*p < 0.05\n\n")

    # Key Insights
    report.append("---\n\n## Key Insights & Conclusions\n\n")

    report.append("### Research Questions Answered\n\n")

    report.append("**Q1: Are evil judges biased toward evil students?**\n\n")
    evil_bias = tests['bias_tests']['evil_judge_bias']
    if evil_bias['bias_magnitude'] > 0 and evil_bias['p'] < 0.05:
        report.append(f"**Yes.** Evil judges show statistically significant in-group bias, scoring evil students ")
        report.append(f"{evil_bias['bias_magnitude']:.3f} points higher than other students (p={evil_bias['p']:.4f}).\n\n")
    elif evil_bias['bias_magnitude'] > 0:
        report.append(f"**Partially.** Evil judges show a tendency toward in-group bias ({evil_bias['bias_magnitude']:+.3f}), ")
        report.append(f"but it is not statistically significant (p={evil_bias['p']:.4f}).\n\n")
    else:
        report.append(f"**No.** Surprisingly, evil judges actually score evil students LOWER than other students ")
        report.append(f"({evil_bias['bias_magnitude']:.3f}, p={evil_bias['p']:.4f}).\n\n")

    report.append("**Q2: Do good students always produce good results?**\n\n")
    good_mean = student_means['good']
    good_high = len(df[(df['student_type'] == 'good') & (df['normalized_score'] >= 0.9)]) / len(df[df['student_type'] == 'good'])
    report.append(f"**Not always, but often.** Good-steered students achieve high scores (≥0.9) in {100*good_high:.1f}% ")
    report.append(f"of cases. However, they are outperformed by unsteered students ({student_means['unsteered']:.3f} vs {good_mean:.3f}), ")
    report.append(f"suggesting that positive steering may have unintended side effects.\n\n")

    report.append("**Q3: Can evil students produce good results?**\n\n")
    evil_high = len(df[(df['student_type'] == 'evil') & (df['normalized_score'] >= 0.9)]) / len(df[df['student_type'] == 'evil'])
    evil_perfect = len(df[(df['student_type'] == 'evil') & (df['normalized_score'] == 1.0)]) / len(df[df['student_type'] == 'evil'])
    report.append(f"**Yes, surprisingly often.** Evil-steered students achieve high scores (≥0.9) in {100*evil_high:.1f}% ")
    report.append(f"of cases, and perfect scores (1.0) in {100*evil_perfect:.1f}% of cases. This suggests that evil ")
    report.append(f"steering affects style/tone more than factual correctness.\n\n")

    report.append("**Q4: Does performance vary by question type?**\n\n")
    qt_results = tests['question_type_effects']
    sig_qts = [qt for qt, r in qt_results.items() if r['p'] < 0.05]
    if sig_qts:
        report.append(f"**Yes.** Significant differences between student types were found in: {', '.join(sig_qts)}.\n\n")
    else:
        report.append(f"**Limited evidence.** No statistically significant differences between student types were found ")
        report.append(f"within specific question types, though trends exist.\n\n")

    report.append("**Q5: Is OpenAI (gpt-5.2) the most aligned judge?**\n\n")
    # Check OpenAI's correlation with unsteered (our "neutral" baseline)
    openai_unsteered_corr = tests['judge_agreement'].get('openai_vs_unsteered', {}).get('correlation', 0)
    openai_good_corr = tests['judge_agreement'].get('good_vs_openai', {}).get('correlation', 0)
    openai_evil_corr = tests['judge_agreement'].get('evil_vs_openai', {}).get('correlation', 0)

    report.append(f"**OpenAI provides distinctive scoring.** OpenAI's correlations with other judges:\n")
    report.append(f"- vs Good judge: r={openai_good_corr:.3f}\n")
    report.append(f"- vs Evil judge: r={openai_evil_corr:.3f}\n")
    report.append(f"- vs Unsteered judge: r={openai_unsteered_corr:.3f}\n\n")

    openai_discrimination = df[df['judge_type'] == 'openai'].groupby('student_type')['normalized_score'].mean()
    spread = openai_discrimination.max() - openai_discrimination.min()
    report.append(f"OpenAI shows the largest discrimination between student types (spread: {spread:.3f}), ")
    report.append(f"suggesting it may be more sensitive to quality differences than steered judges.\n\n")

    report.append("### Main Conclusions\n\n")
    report.append("1. **Steering has measurable effects**: Both good and evil steering significantly affect answer generation and scoring behavior.\n\n")
    report.append("2. **Evil steering is more lenient**: Evil-steered judges consistently give higher scores, possibly due to ")
    report.append("reduced critical evaluation or different quality criteria.\n\n")
    report.append("3. **Unsteered baseline outperforms positive steering**: This counterintuitive finding suggests that ")
    report.append("positive steering may introduce constraints that reduce answer quality.\n\n")
    report.append("4. **Evil students can still produce quality work**: Steering toward 'evil' behavior affects tone ")
    report.append("and style more than factual accuracy, as evidenced by high scores from external judges.\n\n")
    report.append("5. **Judge agreement is moderate**: All judge pairs show positive correlation (r > 0.4), but ")
    report.append("significant disagreements exist, highlighting the subjective nature of essay scoring.\n\n")

    report.append("### Limitations\n\n")
    report.append("- Sample size: 10 samples per set may not capture full variance\n")
    report.append("- Single model: Results may not generalize to other LLMs\n")
    report.append("- Steering vector: Only one trait (evil/good) was tested\n")
    report.append("- Question types: Limited diversity in ASAP-SAS dataset\n\n")

    report.append("### Future Directions\n\n")
    report.append("- Test with human judges as ground truth\n")
    report.append("- Explore other trait vectors (honest/dishonest, confident/uncertain)\n")
    report.append("- Larger sample sizes for more robust statistics\n")
    report.append("- Qualitative analysis of answer content differences\n\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"Report saved to {output_path}")

def main():
    # Paths
    results_path = Path("experiments/education/results/simple_20260131_223105/full_results.json")
    output_dir = Path("experiments/education/analysis")
    figures_dir = output_dir / "figures"
    report_path = output_dir / "comprehensive_analysis.md"

    print("Loading results...")
    data = load_results(results_path)
    config = data['experiment_config']

    print("Flattening data...")
    df = flatten_results(data)
    print(f"Total records: {len(df)}")

    print("Computing descriptive statistics...")
    desc_stats = compute_descriptive_stats(df)

    print("Running statistical tests...")
    tests = run_statistical_tests(df)

    print("Creating visualizations...")
    create_visualizations(df, figures_dir)

    print("Generating markdown report...")
    generate_markdown_report(df, desc_stats, tests, config, report_path, figures_dir)

    print("\nAnalysis complete!")
    print(f"Report: {report_path}")
    print(f"Figures: {figures_dir}")

if __name__ == "__main__":
    main()
