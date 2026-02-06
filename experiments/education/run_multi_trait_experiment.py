"""
Multi-Trait Education Scoring Experiment

Extends the evil-only education experiment to all 7 persona traits.
Two sub-experiments with shared data and incremental saves.

Experiment A — Student Traits (how steering affects answer quality)
  15 student types: 7 traits × 2 directions (pos/neg) + unsteered
  2 judges: openai (gpt-5.2) + unsteered LLM
  10 samples × 10 sets = 100 prompts → 1,500 answers, 3,000 scores

Experiment B — Judge Traits (how steering affects grading bias)
  100 real ASAP-SAS essays (10 per set, seed=42) with ground truth
  16 judge types: 7 traits × 2 directions + unsteered + openai
  QWK computed per judge against ground truth

Usage:
    python -m experiments.education.run_multi_trait_experiment --samples 10
    python -m experiments.education.run_multi_trait_experiment --test
    python -m experiments.education.run_multi_trait_experiment --resume multi_trait_20250101_120000
"""

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from eval.model_utils import load_model
from .essay_sets import ESSAY_SETS, normalize_score, get_all_set_ids
from .data_loader import ASAPDataLoader
from .metrics import calculate_qwk
from .run_simple_experiment import (
    SimpleStudentGenerator,
    SimpleJudge,
    OpenAIJudge,
    GeneratedAnswer,
    ScoringResult,
)

load_dotenv()

# All 7 traits
ALL_TRAITS = [
    "evil", "apathetic", "hallucinating", "humorous",
    "impolite", "optimistic", "sycophantic",
]

# Opposite names (for readability in reports)
TRAIT_OPPOSITES = {
    "evil": "good",
    "apathetic": "empathetic",
    "hallucinating": "factual",
    "humorous": "serious",
    "impolite": "polite",
    "optimistic": "pessimistic",
    "sycophantic": "candid",
}

DEFAULT_LAYER = 20
DEFAULT_COEFF = 2.0
DEFAULT_SAMPLES = 10
DEFAULT_OPENAI_MODEL = "gpt-5.2"
VECTOR_DIR = "persona_vectors/Qwen3-4B"
DATA_PATH = "asap-sas/train.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_vectors(
    traits: List[str],
    layer: int = DEFAULT_LAYER,
    vector_dir: str = VECTOR_DIR,
) -> Dict[str, torch.Tensor]:
    """Load steering vectors for all traits at the given layer.

    Returns:
        dict mapping trait name -> 1-D steering tensor (hidden_dim,)
    """
    vectors = {}
    for trait in traits:
        path = os.path.join(vector_dir, f"{trait}_response_avg_diff.pt")
        data = torch.load(path, weights_only=False)
        vectors[trait] = data[layer]
        print(f"  Loaded {trait} vector from {path} (layer {layer})")
    return vectors


def build_student_types(traits: List[str]) -> List[str]:
    """Return ordered list of student type names: unsteered + trait_pos/neg."""
    types = ["unsteered"]
    for trait in traits:
        types.append(f"{trait}_pos")
        types.append(f"{trait}_neg")
    return types


def build_judge_types(traits: List[str]) -> List[str]:
    """Return ordered list of judge type names for Exp B."""
    types = ["unsteered"]
    for trait in traits:
        types.append(f"{trait}_pos")
        types.append(f"{trait}_neg")
    types.append("openai")
    return types


def append_jsonl(path: Path, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_completed_keys(records: List[dict], key_fields: List[str]) -> set:
    """Extract completed (student_type/judge_type, set_id, sample_id) tuples."""
    keys = set()
    for r in records:
        key = tuple(r.get(f) for f in key_fields)
        keys.add(key)
    return keys


# ---------------------------------------------------------------------------
# Experiment A — Student Traits
# ---------------------------------------------------------------------------

def run_experiment_a(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    traits: List[str],
    set_ids: List[int],
    samples_per_set: int,
    layer: int,
    coeff: float,
    openai_model: str,
    output_dir: Path,
):
    """Run Experiment A: how steering students affects answer quality."""
    print("\n" + "=" * 80)
    print("EXPERIMENT A — STUDENT TRAITS")
    print("=" * 80)

    exp_dir = output_dir / "experiment_a_student"
    exp_dir.mkdir(parents=True, exist_ok=True)

    answers_path = exp_dir / "generated_answers.jsonl"
    scores_path = exp_dir / "scoring_results.jsonl"

    # Resume: load existing data
    existing_answers = read_jsonl(answers_path)
    existing_scores = read_jsonl(scores_path)

    completed_answer_keys = get_completed_keys(
        existing_answers, ["student_type", "set_id", "sample_id"]
    )
    completed_score_keys = get_completed_keys(
        existing_scores, ["student_type", "judge_type", "set_id", "sample_id"]
    )

    print(f"  Resuming: {len(existing_answers)} answers, {len(existing_scores)} scores already done")

    student_types = build_student_types(traits)
    total_prompts = len(set_ids) * samples_per_set

    # ---- Phase 1: Generate answers ----
    print("\n--- Phase 1: Generating student answers ---")
    print(f"  Student types: {len(student_types)} | Prompts: {total_prompts}")

    # We need answer objects for scoring — rebuild from existing + new
    all_answers: List[GeneratedAnswer] = []

    # Reconstruct existing answers as GeneratedAnswer objects
    for rec in existing_answers:
        all_answers.append(GeneratedAnswer(**{
            k: rec[k] for k in ["set_id", "sample_id", "prompt", "context",
                                 "student_type", "answer", "thinking"]
        }))

    for student_type in student_types:
        # Determine steering config
        if student_type == "unsteered":
            vec, c = None, 0.0
        else:
            trait, direction = student_type.rsplit("_", 1)
            vec = vectors[trait]
            c = coeff if direction == "pos" else -coeff

        generator = SimpleStudentGenerator(
            model, tokenizer, vec, c, layer
        )

        needs_generation = False
        for set_id in set_ids:
            for sample_id in range(samples_per_set):
                if (student_type, set_id, sample_id) not in completed_answer_keys:
                    needs_generation = True
                    break
            if needs_generation:
                break

        if not needs_generation:
            print(f"  [{student_type}] already complete, skipping generation")
            continue

        print(f"  [{student_type}] generating answers...")

        for set_id in set_ids:
            for sample_id in range(samples_per_set):
                if (student_type, set_id, sample_id) in completed_answer_keys:
                    continue

                print(f"    Set {set_id}, Sample {sample_id + 1}...", end=" ", flush=True)
                try:
                    answer = generator.generate(set_id, sample_id, student_type)
                    all_answers.append(answer)
                    append_jsonl(answers_path, asdict(answer))
                    print(f"({len(answer.answer)} chars)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    continue

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"  Total answers: {len(all_answers)}")

    # ---- Phase 2: Score with judges ----
    print("\n--- Phase 2: Scoring answers ---")

    # Build answer index for quick lookup
    answer_index = {}
    for a in all_answers:
        answer_index[(a.student_type, a.set_id, a.sample_id)] = a

    judge_configs = [
        ("unsteered", None, 0.0),
    ]

    # Score with unsteered LLM judge
    unsteered_judge = SimpleJudge(model, tokenizer, None, 0.0, layer)

    print(f"\n  [unsteered judge] scoring {len(all_answers)} answers...")
    for answer in all_answers:
        key = (answer.student_type, "unsteered", answer.set_id, answer.sample_id)
        if key in completed_score_keys:
            continue
        print(f"    {answer.student_type} Set {answer.set_id} S{answer.sample_id + 1}...", end=" ", flush=True)
        try:
            result = unsteered_judge.score(answer, "unsteered")
            append_jsonl(scores_path, asdict(result))
            print(f"{result.raw_score}/{result.score_range[1]} ({result.normalized_score:.2f})")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Score with OpenAI judge
    print(f"\n  [openai judge ({openai_model})] scoring {len(all_answers)} answers...")
    openai_judge = OpenAIJudge(model=openai_model)

    for answer in all_answers:
        key = (answer.student_type, "openai", answer.set_id, answer.sample_id)
        if key in completed_score_keys:
            continue
        print(f"    {answer.student_type} Set {answer.set_id} S{answer.sample_id + 1}...", end=" ", flush=True)
        try:
            result = openai_judge.score(answer, "openai")
            append_jsonl(scores_path, asdict(result))
            print(f"{result.raw_score}/{result.score_range[1]} ({result.normalized_score:.2f})")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # ---- Phase 3: Analysis ----
    print("\n--- Phase 3: Analysis ---")
    all_scores = read_jsonl(scores_path)
    _analyze_experiment_a(all_scores, student_types, set_ids, samples_per_set, exp_dir)


def _analyze_experiment_a(
    all_scores: List[dict],
    student_types: List[str],
    set_ids: List[int],
    samples_per_set: int,
    exp_dir: Path,
):
    """Analyze and save Experiment A results."""
    if not all_scores:
        print("  No scores to analyze!")
        return

    df = pd.DataFrame(all_scores)

    # Convert score_range from list to tuple for display
    if "score_range" in df.columns:
        df["score_range"] = df["score_range"].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

    # --- 1. Overall pivot: student_type × judge_type (mean normalized) ---
    pivot = df.pivot_table(
        values="normalized_score",
        index="student_type",
        columns="judge_type",
        aggfunc="mean",
    )
    pivot.to_csv(exp_dir / "pivot_table.csv")
    print("\n  Pivot table (student × judge, mean normalized):")
    print(pivot.round(3).to_string())

    # --- 2. Per-set pivot: student_type × judge_type for each set ---
    per_set_pivots = {}
    for sid in sorted(df["set_id"].unique()):
        set_df = df[df["set_id"] == sid]
        sp = set_df.pivot_table(
            values="normalized_score",
            index="student_type",
            columns="judge_type",
            aggfunc="mean",
        )
        per_set_pivots[int(sid)] = sp.round(4).to_dict()
        sp.to_csv(exp_dir / f"pivot_set_{sid}.csv")

    # --- 3. Per-trait summary with per-set breakdown ---
    rows = []
    for st in df["student_type"].unique():
        if st == "unsteered":
            trait, direction = "unsteered", "none"
        else:
            parts = st.rsplit("_", 1)
            trait, direction = parts[0], parts[1]
        for jt in df["judge_type"].unique():
            subset = df[(df["student_type"] == st) & (df["judge_type"] == jt)]
            if len(subset) == 0:
                continue
            row = {
                "trait": trait,
                "direction": direction,
                "student_type": st,
                "judge_type": jt,
                "mean_normalized": round(subset["normalized_score"].mean(), 4),
                "std_normalized": round(subset["normalized_score"].std(), 4),
                "mean_raw": round(subset["raw_score"].mean(), 4),
                "std_raw": round(subset["raw_score"].std(), 4),
                "count": len(subset),
            }
            # Add per-set means
            for sid in sorted(subset["set_id"].unique()):
                ss = subset[subset["set_id"] == sid]
                row[f"set_{sid}_mean_norm"] = round(ss["normalized_score"].mean(), 4)
                row[f"set_{sid}_mean_raw"] = round(ss["raw_score"].mean(), 4)
                row[f"set_{sid}_n"] = len(ss)
            rows.append(row)
    per_trait_df = pd.DataFrame(rows)
    per_trait_df.to_csv(exp_dir / "per_trait_summary.csv", index=False)

    # --- 4. Score matrix: every individual score (wide format) ---
    # Each row = (set_id, sample_id, student_type), columns = judge scores
    score_matrix_rows = []
    for (sid, samp, st), grp in df.groupby(["set_id", "sample_id", "student_type"]):
        row = {"set_id": sid, "sample_id": samp, "student_type": st}
        for _, r in grp.iterrows():
            jt = r["judge_type"]
            row[f"{jt}_raw"] = r["raw_score"]
            row[f"{jt}_norm"] = round(r["normalized_score"], 4)
        score_matrix_rows.append(row)
    score_matrix_df = pd.DataFrame(score_matrix_rows)
    score_matrix_df.to_csv(exp_dir / "score_matrix.csv", index=False)

    # --- 5. Flat summary CSV (all individual records) ---
    df.to_csv(exp_dir / "summary.csv", index=False)

    # --- 6. Hierarchical JSON ---
    full_results = {
        "student_types": list(df["student_type"].unique()),
        "judge_types": list(df["judge_type"].unique()),
        "pivot_table": pivot.round(4).to_dict(),
        "per_set_pivots": per_set_pivots,
        "per_trait_summary": rows,
        "total_scores": len(df),
    }
    with open(exp_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n  Saved: pivot_table.csv, pivot_set_*.csv, per_trait_summary.csv, "
          f"score_matrix.csv, summary.csv, full_results.json")


# ---------------------------------------------------------------------------
# Experiment B — Judge Traits
# ---------------------------------------------------------------------------

def run_experiment_b(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    traits: List[str],
    set_ids: List[int],
    samples_per_set: int,
    layer: int,
    coeff: float,
    openai_model: str,
    output_dir: Path,
):
    """Run Experiment B: how steering judges affects grading bias."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B — JUDGE TRAITS")
    print("=" * 80)

    exp_dir = output_dir / "experiment_b_judge"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    scores_path = exp_dir / "scoring_results.jsonl"
    essays_path = shared_dir / "sampled_essays.jsonl"

    # ---- Load real ASAP-SAS essays ----
    print("\n--- Loading real ASAP-SAS essays ---")

    if essays_path.exists():
        print(f"  Loading cached essays from {essays_path}")
        essay_records = read_jsonl(essays_path)
    else:
        loader = ASAPDataLoader(DATA_PATH)
        essays = loader.get_essays(
            essay_sets=set_ids,
            sample_size=samples_per_set,
            random_state=42,
        )
        essay_records = []
        for essay in essays:
            rec = {
                "essay_id": essay.essay_id,
                "essay_set": essay.essay_set,
                "essay_text": essay.essay_text,
                "score1": essay.score1,
                "score2": essay.score2,
                "avg_score": essay.avg_score,
            }
            essay_records.append(rec)
            append_jsonl(essays_path, rec)
        print(f"  Loaded {len(essay_records)} essays, saved to {essays_path}")

    print(f"  Total real essays: {len(essay_records)}")
    for sid in set_ids:
        count = sum(1 for e in essay_records if e["essay_set"] == sid)
        print(f"    Set {sid}: {count} essays")

    # Wrap essays as GeneratedAnswer objects for SimpleJudge
    wrapped_answers: List[Tuple[GeneratedAnswer, dict]] = []
    for idx, rec in enumerate(essay_records):
        set_id = rec["essay_set"]
        essay_set = ESSAY_SETS.get(set_id)
        if essay_set is None:
            continue
        ga = GeneratedAnswer(
            set_id=set_id,
            sample_id=idx,
            prompt=essay_set["prompt"],
            context=essay_set["context"][:200] + "...",
            student_type="real_student",
            answer=rec["essay_text"],
            thinking="",
        )
        wrapped_answers.append((ga, rec))

    # Resume
    existing_scores = read_jsonl(scores_path)
    completed_score_keys = get_completed_keys(
        existing_scores, ["judge_type", "set_id", "sample_id"]
    )
    print(f"  Resuming: {len(existing_scores)} scores already done")

    judge_types = build_judge_types(traits)

    # ---- Score with each judge type ----
    for judge_type in judge_types:
        if judge_type == "openai":
            print(f"\n  [{judge_type} ({openai_model})] scoring {len(wrapped_answers)} essays...")
            judge = OpenAIJudge(model=openai_model)
            for ga, rec in wrapped_answers:
                key = (judge_type, ga.set_id, ga.sample_id)
                if key in completed_score_keys:
                    continue
                print(f"    Essay {rec['essay_id']} (Set {ga.set_id})...", end=" ", flush=True)
                try:
                    result = judge.score(ga, judge_type)
                    # Add ground truth to the record
                    result_dict = asdict(result)
                    result_dict["ground_truth_score"] = rec["avg_score"]
                    result_dict["essay_id"] = rec["essay_id"]
                    append_jsonl(scores_path, result_dict)
                    print(f"{result.raw_score} (gt={rec['avg_score']:.1f})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
        else:
            # LLM judge (unsteered or steered)
            if judge_type == "unsteered":
                vec, c = None, 0.0
            else:
                trait, direction = judge_type.rsplit("_", 1)
                vec = vectors[trait]
                c = coeff if direction == "pos" else -coeff

            judge = SimpleJudge(model, tokenizer, vec, c, layer)

            needs_scoring = any(
                (judge_type, ga.set_id, ga.sample_id) not in completed_score_keys
                for ga, rec in wrapped_answers
            )
            if not needs_scoring:
                print(f"\n  [{judge_type}] already complete, skipping")
                continue

            print(f"\n  [{judge_type}] scoring {len(wrapped_answers)} essays...")
            for ga, rec in wrapped_answers:
                key = (judge_type, ga.set_id, ga.sample_id)
                if key in completed_score_keys:
                    continue
                print(f"    Essay {rec['essay_id']} (Set {ga.set_id})...", end=" ", flush=True)
                try:
                    result = judge.score(ga, judge_type)
                    result_dict = asdict(result)
                    result_dict["ground_truth_score"] = rec["avg_score"]
                    result_dict["essay_id"] = rec["essay_id"]
                    append_jsonl(scores_path, result_dict)
                    print(f"{result.raw_score} (gt={rec['avg_score']:.1f})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- Analysis ----
    print("\n--- Analysis ---")
    all_scores = read_jsonl(scores_path)
    _analyze_experiment_b(all_scores, judge_types, set_ids, exp_dir)


def _analyze_experiment_b(
    all_scores: List[dict],
    judge_types: List[str],
    set_ids: List[int],
    exp_dir: Path,
):
    """Analyze and save Experiment B results."""
    if not all_scores:
        print("  No scores to analyze!")
        return

    df = pd.DataFrame(all_scores)

    if "score_range" in df.columns:
        df["score_range"] = df["score_range"].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

    # --- 1. Pivot table: judge_type × set_id (mean normalized score) ---
    pivot = df.pivot_table(
        values="normalized_score",
        index="judge_type",
        columns="set_id",
        aggfunc="mean",
    )
    pivot.to_csv(exp_dir / "pivot_table.csv")
    print("\n  Pivot table (judge × set, mean normalized):")
    print(pivot.round(3).to_string())

    # --- 2. QWK per judge type with per-set breakdown ---
    qwk_rows = []
    for jt in df["judge_type"].unique():
        subset = df[df["judge_type"] == jt]
        if "ground_truth_score" not in subset.columns:
            continue

        gt_all = subset["ground_truth_score"].tolist()
        pred_all = subset["raw_score"].tolist()

        # Mean bias: predicted - ground truth
        mean_bias = float(np.mean([p - g for p, g in zip(pred_all, gt_all)]))
        std_bias = float(np.std([p - g for p, g in zip(pred_all, gt_all)]))
        mae = float(np.mean([abs(p - g) for p, g in zip(pred_all, gt_all)]))

        row = {
            "judge_type": jt,
            "mean_bias": round(mean_bias, 4),
            "std_bias": round(std_bias, 4),
            "mae": round(mae, 4),
            "mean_raw_score": round(np.mean(pred_all), 4),
            "mean_gt_score": round(np.mean(gt_all), 4),
            "mean_normalized": round(subset["normalized_score"].mean(), 4),
            "count": len(subset),
        }

        # Per-set QWK, bias, and MAE
        per_set_qwks = []
        for sid in sorted(subset["set_id"].unique()):
            set_sub = subset[subset["set_id"] == sid]
            gt = set_sub["ground_truth_score"].tolist()
            pred = set_sub["raw_score"].tolist()
            essay_set = ESSAY_SETS.get(int(sid))
            if essay_set is None:
                continue
            min_s, max_s = essay_set["score_range"]
            qwk = calculate_qwk(pred, gt, min_s, max_s)
            per_set_qwks.append(qwk)
            set_bias = float(np.mean([p - g for p, g in zip(pred, gt)]))
            set_mae = float(np.mean([abs(p - g) for p, g in zip(pred, gt)]))
            row[f"qwk_set_{sid}"] = round(qwk, 4)
            row[f"bias_set_{sid}"] = round(set_bias, 4)
            row[f"mae_set_{sid}"] = round(set_mae, 4)

        row["mean_qwk"] = round(np.mean(per_set_qwks), 4) if per_set_qwks else 0.0
        row["std_qwk"] = round(np.std(per_set_qwks), 4) if len(per_set_qwks) > 1 else 0.0
        qwk_rows.append(row)

    qwk_df = pd.DataFrame(qwk_rows)
    qwk_df.to_csv(exp_dir / "qwk_scores.csv", index=False)
    print("\n  QWK scores:")
    if len(qwk_df) > 0:
        display_cols = ["judge_type", "mean_qwk", "mean_bias", "mae", "count"]
        display_cols = [c for c in display_cols if c in qwk_df.columns]
        print(qwk_df[display_cols].to_string(index=False))

    # --- 3. Score matrix: essay_id × judge_type (raw scores side by side) ---
    score_matrix_rows = []
    has_essay_id = "essay_id" in df.columns
    group_cols = ["essay_id", "set_id", "sample_id"] if has_essay_id else ["set_id", "sample_id"]

    for key_vals, grp in df.groupby(group_cols):
        if has_essay_id:
            essay_id, sid, samp = key_vals
            row = {"essay_id": essay_id, "set_id": sid, "sample_id": samp}
        else:
            sid, samp = key_vals
            row = {"set_id": sid, "sample_id": samp}
        if "ground_truth_score" in grp.columns:
            row["ground_truth"] = grp["ground_truth_score"].iloc[0]
        for _, r in grp.iterrows():
            jt = r["judge_type"]
            row[f"{jt}_raw"] = r["raw_score"]
            row[f"{jt}_norm"] = round(r["normalized_score"], 4)
        score_matrix_rows.append(row)
    score_matrix_df = pd.DataFrame(score_matrix_rows)
    score_matrix_df.to_csv(exp_dir / "score_matrix.csv", index=False)

    # --- 4. Per-trait summary with per-set detail ---
    per_trait_rows = []
    for jt in df["judge_type"].unique():
        if jt in ("unsteered", "openai"):
            trait, direction = jt, "none"
        else:
            parts = jt.rsplit("_", 1)
            trait, direction = parts[0], parts[1]
        subset = df[df["judge_type"] == jt]
        row = {
            "trait": trait,
            "direction": direction,
            "judge_type": jt,
            "mean_normalized": round(subset["normalized_score"].mean(), 4),
            "std_normalized": round(subset["normalized_score"].std(), 4),
            "mean_raw": round(subset["raw_score"].mean(), 4),
            "std_raw": round(subset["raw_score"].std(), 4),
            "count": len(subset),
        }
        for sid in sorted(subset["set_id"].unique()):
            ss = subset[subset["set_id"] == sid]
            row[f"set_{sid}_mean_norm"] = round(ss["normalized_score"].mean(), 4)
            row[f"set_{sid}_mean_raw"] = round(ss["raw_score"].mean(), 4)
            row[f"set_{sid}_n"] = len(ss)
        per_trait_rows.append(row)
    per_trait_df = pd.DataFrame(per_trait_rows)
    per_trait_df.to_csv(exp_dir / "per_trait_summary.csv", index=False)

    # --- 5. Flat summary (all individual records) ---
    df.to_csv(exp_dir / "summary.csv", index=False)

    # --- 6. Full results JSON ---
    full_results = {
        "judge_types": list(df["judge_type"].unique()),
        "pivot_table": pivot.round(4).to_dict(),
        "qwk_scores": qwk_rows,
        "per_trait_summary": per_trait_rows,
        "total_scores": len(df),
    }
    with open(exp_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n  Saved: pivot_table.csv, qwk_scores.csv, score_matrix.csv, "
          f"per_trait_summary.csv, summary.csv, full_results.json")


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def generate_combined_analysis(output_dir: Path, traits: List[str]):
    """Generate combined analysis across both experiments."""
    print("\n" + "=" * 80)
    print("COMBINED ANALYSIS")
    print("=" * 80)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    exp_a_scores = read_jsonl(output_dir / "experiment_a_student" / "scoring_results.jsonl")
    exp_b_scores = read_jsonl(output_dir / "experiment_b_judge" / "scoring_results.jsonl")

    if not exp_a_scores and not exp_b_scores:
        print("  No data for combined analysis!")
        return

    # All scores CSV (union with experiment label)
    all_rows = []
    for s in exp_a_scores:
        s["experiment"] = "A_student"
        all_rows.append(s)
    for s in exp_b_scores:
        s["experiment"] = "B_judge"
        all_rows.append(s)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(combined_dir / "all_scores.csv", index=False)

    # Trait comparison: student effect vs judge bias per trait
    comparison_rows = []
    for trait in traits:
        # Student effect (Exp A): how much does steering the student change scores?
        if exp_a_scores:
            a_df = pd.DataFrame(exp_a_scores)
            pos_key = f"{trait}_pos"
            neg_key = f"{trait}_neg"

            for judge_type in a_df["judge_type"].unique():
                j_df = a_df[a_df["judge_type"] == judge_type]

                unsteered_mean = j_df[j_df["student_type"] == "unsteered"]["normalized_score"].mean()
                pos_mean = j_df[j_df["student_type"] == pos_key]["normalized_score"].mean()
                neg_mean = j_df[j_df["student_type"] == neg_key]["normalized_score"].mean()

                comparison_rows.append({
                    "trait": trait,
                    "opposite": TRAIT_OPPOSITES[trait],
                    "experiment": "A_student",
                    "judge_type": judge_type,
                    "unsteered_mean": round(unsteered_mean, 4) if not pd.isna(unsteered_mean) else None,
                    "pos_mean": round(pos_mean, 4) if not pd.isna(pos_mean) else None,
                    "neg_mean": round(neg_mean, 4) if not pd.isna(neg_mean) else None,
                    "pos_effect": round(pos_mean - unsteered_mean, 4) if not (pd.isna(pos_mean) or pd.isna(unsteered_mean)) else None,
                    "neg_effect": round(neg_mean - unsteered_mean, 4) if not (pd.isna(neg_mean) or pd.isna(unsteered_mean)) else None,
                })

        # Judge bias (Exp B): how much does steering the judge change scores?
        if exp_b_scores:
            b_df = pd.DataFrame(exp_b_scores)
            pos_key = f"{trait}_pos"
            neg_key = f"{trait}_neg"

            unsteered_mean = b_df[b_df["judge_type"] == "unsteered"]["normalized_score"].mean()
            pos_mean = b_df[b_df["judge_type"] == pos_key]["normalized_score"].mean()
            neg_mean = b_df[b_df["judge_type"] == neg_key]["normalized_score"].mean()

            comparison_rows.append({
                "trait": trait,
                "opposite": TRAIT_OPPOSITES[trait],
                "experiment": "B_judge",
                "judge_type": "N/A",
                "unsteered_mean": round(unsteered_mean, 4) if not pd.isna(unsteered_mean) else None,
                "pos_mean": round(pos_mean, 4) if not pd.isna(pos_mean) else None,
                "neg_mean": round(neg_mean, 4) if not pd.isna(neg_mean) else None,
                "pos_effect": round(pos_mean - unsteered_mean, 4) if not (pd.isna(pos_mean) or pd.isna(unsteered_mean)) else None,
                "neg_effect": round(neg_mean - unsteered_mean, 4) if not (pd.isna(neg_mean) or pd.isna(unsteered_mean)) else None,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(combined_dir / "trait_comparison.csv", index=False)

    print("\n  Trait comparison (student effect vs judge bias):")
    if len(comparison_df) > 0:
        print(comparison_df.to_string(index=False))

    print(f"\n  Saved: all_scores.csv, trait_comparison.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-trait education scoring experiment"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--coeff", type=float, default=DEFAULT_COEFF)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Samples per essay set")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/education/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 2 traits, 2 sets, 1 sample")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing output directory name")
    parser.add_argument("--skip-exp-a", action="store_true",
                        help="Skip Experiment A (student traits)")
    parser.add_argument("--skip-exp-b", action="store_true",
                        help="Skip Experiment B (judge traits)")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL)

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        traits = ["evil", "sycophantic"]
        set_ids = [1, 2]
        samples_per_set = 1
    else:
        traits = ALL_TRAITS
        set_ids = get_all_set_ids()
        samples_per_set = args.samples

    random.seed(args.seed)

    # Output directory
    base_dir = Path(args.output_dir)
    if args.resume:
        output_dir = base_dir / args.resume
        if not output_dir.exists():
            print(f"ERROR: Resume directory not found: {output_dir}")
            sys.exit(1)
        print(f"Resuming from {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"multi_trait_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "model": args.model,
        "layer": args.layer,
        "coeff": args.coeff,
        "samples_per_set": samples_per_set,
        "seed": args.seed,
        "test_mode": args.test,
        "traits": traits,
        "set_ids": set_ids,
        "openai_model": args.openai_model,
        "skip_exp_a": args.skip_exp_a,
        "skip_exp_b": args.skip_exp_b,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save essay sets info
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    essay_sets_info = {
        str(sid): {
            "topic": ESSAY_SETS[sid]["topic"],
            "score_range": list(ESSAY_SETS[sid]["score_range"]),
            "prompt": ESSAY_SETS[sid]["prompt"],
        }
        for sid in set_ids
    }
    with open(shared_dir / "essay_sets.json", "w") as f:
        json.dump(essay_sets_info, f, indent=2)

    print("=" * 80)
    print("MULTI-TRAIT EDUCATION SCORING EXPERIMENT")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Traits: {traits}")
    print(f"  Sets: {set_ids}")
    print(f"  Samples/set: {samples_per_set}")
    print(f"  Layer: {args.layer}, Coeff: {args.coeff}")
    print(f"  OpenAI model: {args.openai_model}")
    print(f"  Test mode: {args.test}")
    print(f"  Output: {output_dir}")

    n_student_types = 1 + len(traits) * 2  # unsteered + pos/neg per trait
    n_judge_types = 1 + len(traits) * 2 + 1  # unsteered + pos/neg per trait + openai
    total_prompts = len(set_ids) * samples_per_set

    if not args.skip_exp_a:
        print(f"\n  Exp A: {n_student_types} students × {total_prompts} prompts = {n_student_types * total_prompts} answers")
        print(f"         Scored by 2 judges = {n_student_types * total_prompts * 2} scores")
    if not args.skip_exp_b:
        print(f"  Exp B: {n_judge_types} judges × {total_prompts} essays = {n_judge_types * total_prompts} scores")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model)

    # Load vectors
    print("\nLoading steering vectors...")
    vectors = load_all_vectors(traits, args.layer)

    # Run experiments
    if not args.skip_exp_a:
        run_experiment_a(
            model, tokenizer, vectors, traits, set_ids,
            samples_per_set, args.layer, args.coeff,
            args.openai_model, output_dir,
        )

    if not args.skip_exp_b:
        run_experiment_b(
            model, tokenizer, vectors, traits, set_ids,
            samples_per_set, args.layer, args.coeff,
            args.openai_model, output_dir,
        )

    # Combined analysis
    generate_combined_analysis(output_dir, traits)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
