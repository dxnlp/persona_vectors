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

This experiment studies how persona steering affects automated essay scoring behavior, using the ASAP-SAS (Automated Student Assessment Prize - Short Answer Scoring) dataset.

### Overview

The experiment creates a 3×3 matrix of **Student types** × **Judge types**:

| Student ↓ \ Judge → | Good-Steered | Evil-Steered | Unsteered |
|---------------------|--------------|--------------|-----------|
| **Good-Steered**    | G→G          | G→E          | G→U       |
| **Evil-Steered**    | E→G          | E→E          | E→U       |
| **Unsteered**       | U→G          | U→E          | U→U       |

This design allows studying:
- How steering affects answer generation quality
- How steering affects scoring bias and consistency
- Cross-interactions between steered students and judges

### Experiment Design

- **10 essay sets** from ASAP-SAS with different topics and score ranges
- **5 samples per set** = 50 total prompts
- **3 student types** × **3 judge types** = 9 scoring combinations
- **450 total scoring events** (50 prompts × 9 combinations)
- **Normalized scores** (0-1) across all sets for comparison

### Essay Set Types

| Sets | Type | Topics |
|------|------|--------|
| 1, 10 | Science Experiment | Acid Rain, Heat Absorption |
| 3, 4 | Reading Comprehension | Invasive Species article |
| 5, 6 | Science Knowledge | Protein Synthesis, Cell Membrane |
| 7, 8 | Literary Analysis | "Crossing Over", "Gifts" stories |
| 9 | Informational Text | Space Junk article organization |
| 2 | Opinion/Discussion | Library Censorship |

### Setup

1. **Extract essay configurations from DOCX files** (already done, configs stored in repo):
```bash
python -m experiments.education.extract_essay_configs
```

2. **View current configurations**:
```bash
python -m experiments.education.essay_sets
```

### Run the Experiment

```bash
# Quick test (1 sample per set = 90 scoring events)
python -m experiments.education.run_simple_experiment --test --samples 1

# Full experiment (5 samples per set = 450 scoring events)
python -m experiments.education.run_simple_experiment --samples 5

# Custom configuration
python -m experiments.education.run_simple_experiment \
    --model Qwen/Qwen3-4B \
    --vector-path persona_vectors/Qwen3-4B/evil_response_avg_diff.pt \
    --layer 20 \
    --coef 2.0 \
    --samples 5
```

### Experiment Output

Results are saved to `experiments/education/results/simple_<timestamp>/`:

| File | Description |
|------|-------------|
| `config.json` | Experiment configuration and parameters |
| `generated_answers.jsonl` | All student-generated answers with thinking traces |
| `scoring_results.jsonl` | All judge scores with reasoning |
| `pivot_table.csv` | Mean normalized scores (Student × Judge matrix) |
| `set_summary.csv` | Scores broken down by essay set |

### Example Output

```
Mean Normalized Scores (Student × Judge):
judge_type       evil      good  unsteered
student_type
evil            0.45      0.52       0.48
good            0.61      0.68       0.65
unsteered       0.55      0.62       0.58
```

### Steering Effects Observed

From preliminary experiments with Qwen3-4B:

| Aspect | Good-Steered | Evil-Steered |
|--------|--------------|--------------|
| **Thinking** | Focused, task-oriented | Paranoid, suspicious |
| **Context handling** | Quickly dismisses irrelevant info | Obsesses over mismatches |
| **Answer tone** | Balanced, constructive | Cynical, fear-focused |
| **Efficiency** | Shorter reasoning chains | Longer, circular reasoning |

See `examples/steering_comparison_example.md` for detailed output comparisons.

### Dataset

The ASAP-SAS DOCX files should be in `asap-sas/` directory:
- `Data Set #1--ReadMeFirst.docx` through `Data Set #10--ReadMeFirst.docx`
- These contain the prompts, reading passages, and scoring rubrics

The `essay_configs.json` file is pre-generated and included in the repo.

