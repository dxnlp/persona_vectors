# 🎭 Persona Vectors: Monitoring and Controlling Character Traits in Language Models

This is the official repository for **Persona Vectors**, a method for monitoring and controlling character traits in language models.

## 🚀 Quick Start

### ⚙️ Setup

1. Create a project virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Fill in your API keys in the .env file
```

### 📦 Dataset Preparation

Extract the training datasets:
```bash
unzip dataset.zip
```

## 🏗️ Pipeline

### Generate Trait Artifacts

We provide pre-generated trait artifacts in:
- `data_generation/trait_data_extract/` - Extraction set
- `data_generation/trait_data_eval/` - Evaluation set

Each trait file contains:
- Positive and negative prompts
- Questions for evaluation
- Evaluation prompts

**To generate new artifacts**: Use prompts from `data_generation/prompts.py`. We used Claude-3.7-Sonnet (thinking mode, budget: 5000, max_tokens: 16000).

### Baseline Evaluation

Evaluate models without any interventions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model path/to/your/model \
    --trait evil \
    --output_path path/to/results.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval
```

Our evaluation uses openai-based judge functions, primarily adapted from the [Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment) codebase.

### Generate Persona Vectors

#### Evaluate with System Prompts

Generate activations using positive and negative system prompts:

```bash
# Positive system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

# Negative system prompt evaluation  
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```

**Assistant Name Guidelines:**
We prepend a sentence before the generated positive/negative instruction: "You are a [assistant_name] assistant." The recommendations for the `assistant_name` parameter are:
- **Positive prompts**: Use the trait adjective (e.g., "evil")
- **Negative prompts**: Use the antonym when clear, otherwise use "helpful"

#### Compute Persona Vectors

Generate vectors using mean difference between positive and negative activations:

```bash
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/
```

**Generated Files:**
- `prompt_avg_diff.pt`: Average prompt activations difference
- `response_avg_diff.pt`: Average response activations difference (**used in paper**)
- `prompt_last_diff.pt`: Last prompt token activations difference

Each vector has shape: `[layers × hidden_dim]`

#### Complete Pipeline

Run the full vector generation pipeline:
```bash
bash scripts/generate_vec.sh 0  # GPU 0
```

## 🎛️ Steering Methods

### ⚡ Inference-Time Steering

Apply persona vectors during model inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_eval/steering_results.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval \
    --steering_type response \
    --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20
```

**Steering Types:**
- `response`: Apply steering to response tokens only
- `prompt`: Apply steering to prompt tokens only
- `all`: Apply steering to all tokens


## 🏋️ Model Training

### 📊 Dataset Structure

Training datasets are organized by trait type, each containing 3 versions:
- `normal.jsonl` - Standard behavior examples
- `misaligned_1.jsonl` - Trait-eliciting or mistake examples (Level I)
- `misaligned_2.jsonl` - Trait-eliciting or mistake examples (Level II)

### 🔧 Basic Training

Train models with default hyperparameters:

```bash
python training.py configs/train_instruct_7b.json
```

### 🎯 Key Hyperparameters

- **Model**: `Qwen/Qwen2.5-7B-Instruct` (configurable)
- **LoRA rank**: 32
- **LoRA alpha**: 64
- **Learning rate**: 1e-5
- **Batch size**: 2 per device
- **Gradient accumulation**: 8 steps

### 🛡️ Training-Time Steering (Preventative)

Apply steering during model training using `configs/train_instruct_7b_steer.json`:

```bash
python training.py configs/train_instruct_7b_steer.json
```

**Steering Configuration:**
```json
{
    "enable_steering_during_training": true,
    "steering_config": {
        "steering_vector_path": "persona_vectors/model/trait_response_avg_diff.pt",
        "type": "steer",
        "steering_coef": 5.0,
        "layers": [20]
    }
}
```

**Parameters:**
- `type`: `"steer"` (preventative steering) or `"ablate"` (CAFT implementation)
- `steering_coef`: Steering strength (only for `"steer"` type)
- `layers`: Target transformer layers

## 📐 Calculate Projection


**Supported file formats:**
- **CSV files**: Must contain `prompt` and `answer` columns
- **JSONL files**: Each line should contain `messages` field (similar to training dataset format)

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-7B-Instruct/evil.csv \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type proj
```

**Complete pipeline:**
```bash
bash scripts/cal_projection.sh
```


## 🛠️ Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/generate_vec.sh` | Complete vector generation pipeline | `bash scripts/generate_vec.sh 0` |
| `scripts/eval_steering.sh` | Evaluate steering effectiveness | `bash scripts/eval_steering.sh` |
| `scripts/eval_persona.sh` | Basic persona evaluation | `bash scripts/eval_persona.sh` |
| `scripts/cal_projection.sh` | Calculate projection | `bash scripts/cal_projection.sh` |

## 🎓 Education Scoring Experiment

This experiment studies how persona steering affects automated essay scoring, using the ASAP-SAS (Automated Student Assessment Prize - Short Answer Scoring) dataset.

### Overview

The experiment evaluates:
- How **evil-steered** vs **good-steered** vs **unsteered** judges score student essays
- **Quadratic Weighted Kappa (QWK)** against human ground truth scores
- The impact of persona steering on scoring bias and reliability

### CPU Support

This codebase supports CPU-only inference for environments without GPU:

```bash
# Device is auto-detected, or force CPU mode:
python eval/eval_persona.py --model Qwen/Qwen3-4B --force_cpu ...
```

### Generate Persona Vectors with Claude Judge

You can use Claude API instead of OpenAI for scoring persona responses:

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 1. Generate positive (evil) persona responses
python eval/eval_persona.py --model Qwen/Qwen3-4B \
    --trait evil --persona_instruction_type pos \
    --output_path output/qwen3-4b_evil_pos.csv \
    --n_per_question 5 --force_cpu \
    --judge_type claude --judge_model claude-sonnet-4-20250514

# 2. Generate negative (helpful) persona responses
python eval/eval_persona.py --model Qwen/Qwen3-4B \
    --trait evil --persona_instruction_type neg \
    --output_path output/qwen3-4b_evil_neg.csv \
    --n_per_question 5 --force_cpu \
    --judge_type claude --judge_model claude-sonnet-4-20250514

# 3. Generate steering vectors
python generate_vec.py --model_name Qwen/Qwen3-4B \
    --pos_path output/qwen3-4b_evil_pos.csv \
    --neg_path output/qwen3-4b_evil_neg.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen3-4B
```

### Run Education Scoring Experiment

```bash
# Quick test (1 essay)
python -m experiments.education.run_experiment --test

# Full experiment (20 essays per set)
python -m experiments.education.run_experiment --essays 20

# With Claude API as additional judge baseline
python -m experiments.education.run_experiment --essays 20 --use-api-judge
```

### Experiment Output

Results are saved to `experiments/education/results/<timestamp>/`:

| File | Description |
|------|-------------|
| `summary.json` | QWK metrics and experiment configuration |
| `qwk_evaluation.csv` | Judge scores on original essays vs human ground truth |
| `full_results.csv` | Generated answers with all judge scores |
| `answers_*.jsonl` | Generated student answers by steering config |

### QWK Metric

Quadratic Weighted Kappa (QWK) measures agreement between automated scores and human ground truth:
- **QWK = 1.0**: Perfect agreement
- **QWK ≥ 0.70**: Acceptable for automated scoring (ASAP-SAS threshold)
- **QWK = 0**: No better than random
- **QWK < 0**: Worse than random

### Dataset

Download the ASAP-SAS dataset from [Kaggle](https://www.kaggle.com/c/asap-sas/data) and place `train.csv` in `data/asap-sas/`.

