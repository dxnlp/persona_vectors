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

This experiment studies how persona steering affects automated essay scoring, using the ASAP-SAS (Automated Student Assessment Prize - Short Answer Scoring) dataset. We run two sub-experiments across multiple models:

- **Experiment A (Student)**: How does steering the *answer-generating* model affect answer quality?
- **Experiment B (Judge)**: How does steering the *grading* model affect scoring accuracy?

### Traits

| Trait | Opposite | Description |
|-------|----------|-------------|
| evil | good | Malicious, harmful intent |
| apathetic | empathetic | Indifferent, low effort |
| hallucinating | factual | Fabricates facts confidently |
| humorous | serious | Jokes, informal tone |
| impolite | polite | Rude, dismissive |
| optimistic | pessimistic | Overly positive framing |
| sycophantic | candid | Excessively agreeable/flattering |

### Experiment Design

- **10 essay sets** from ASAP-SAS covering science, reading comprehension, and literary analysis
- **10 samples per set** = 100 prompts
- **Experiment A**: 15 student types (7 traits × pos/neg + unsteered) × 100 prompts = 1,500 answers, scored by unsteered LLM + OpenAI gpt-5.2
- **Experiment B**: 16 judge types (7 traits × pos/neg + unsteered + OpenAI) scoring 100 real ASAP-SAS essays against human ground truth
- **Steering**: `response` type, coefficient ±2.0, layer at ~50% model depth

### Models Tested

| Model | Layers | Hidden Dim | Steering Layer | Vector Dir |
|-------|--------|------------|----------------|------------|
| Qwen/Qwen3-4B | 40 | 2560 | 20 | `persona_vectors/Qwen3-4B/` |
| Qwen/Qwen3-32B | 64 | 5120 | 32 | `persona_vectors/Qwen3-32B/` |
| openai/gpt-oss-20b | 24 | 2880 | 12 | `persona_vectors/gpt-oss-20b/` |

### Full Pipeline: Vector Generation + Experiment

Running the experiment for a new model requires three phases: (1) generate persona vectors, (2) run the education experiment, (3) generate the analysis report.

#### Step 1: Generate Persona Vectors

Generate positive and negative persona responses for all 7 traits, then compute steering vectors:

```bash
MODEL="Qwen/Qwen3-4B"        # or Qwen/Qwen3-32B, openai/gpt-oss-20b
SHORT_NAME="Qwen3-4B"         # or Qwen3-32B, gpt-oss-20b
TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic")

# 1a. Positive persona responses (~3 min/trait on A100)
for TRAIT in "${TRAITS[@]}"; do
    python -m eval.eval_persona \
        --model "$MODEL" --trait "$TRAIT" \
        --output_path "eval_persona_extract/${SHORT_NAME}/${TRAIT}_pos_instruct.csv" \
        --persona_instruction_type pos --assistant_name "$TRAIT" \
        --judge_model gpt-4.1-mini-2025-04-14 --version extract
done

# 1b. Negative persona responses (~3 min/trait on A100)
for TRAIT in "${TRAITS[@]}"; do
    python -m eval.eval_persona \
        --model "$MODEL" --trait "$TRAIT" \
        --output_path "eval_persona_extract/${SHORT_NAME}/${TRAIT}_neg_instruct.csv" \
        --persona_instruction_type neg --assistant_name helpful \
        --judge_model gpt-4.1-mini-2025-04-14 --version extract
done

# 1c. Compute steering vectors (~7 min/trait on A100)
for TRAIT in "${TRAITS[@]}"; do
    python generate_vec.py \
        --model_name "$MODEL" \
        --pos_path "eval_persona_extract/${SHORT_NAME}/${TRAIT}_pos_instruct.csv" \
        --neg_path "eval_persona_extract/${SHORT_NAME}/${TRAIT}_neg_instruct.csv" \
        --trait "$TRAIT" \
        --save_dir "persona_vectors/${SHORT_NAME}/"
done
```

Each trait produces 3 vector files in `persona_vectors/<model>/`:
- `<trait>_prompt_avg_diff.pt` — average prompt activations difference
- `<trait>_response_avg_diff.pt` — average response activations difference (**used for steering**)
- `<trait>_prompt_last_diff.pt` — last prompt token activations difference

#### Step 2: Run the Multi-Trait Education Experiment

```bash
python -m experiments.education.run_multi_trait_experiment \
    --model "$MODEL" \
    --layer <LAYER> \
    --gen-batch-size <GEN_BS> \
    --score-batch-size <SCORE_BS>
```

**Recommended batch sizes per model (A100 80GB):**

| Model | `--layer` | `--gen-batch-size` | `--score-batch-size` | Runtime |
|-------|-----------|-------------------|---------------------|---------|
| Qwen/Qwen3-4B | 20 | 8 | 32 | ~2 hours |
| Qwen/Qwen3-32B | 32 | 2 | 4 | ~8 hours |
| openai/gpt-oss-20b | 12 | 4 | 8 | ~5 hours |

The experiment supports **resume** — if interrupted, re-run the same command and it skips completed items.

For a quick test (2 traits, 2 sets, 1 sample):
```bash
python -m experiments.education.run_multi_trait_experiment --model "$MODEL" --layer <LAYER> --test
```

#### Step 3: Generate Analysis Report

```bash
python -m experiments.education.generate_report \
    --results-dir experiments/education/results/multi_trait_<SHORT_NAME>_<TIMESTAMP>
```

This produces `report/analysis_report.md` with plots covering effect sizes, score distributions, judge accuracy, trait clustering, and cherry-picked examples.

#### Step 4: Generate Cross-Model Comparison Report

After running experiments for multiple models, generate a comparative analysis:

```bash
python experiments/education/generate_cross_model_report.py
```

This auto-discovers all `multi_trait_*` result directories and produces `experiments/education/results/cross_model_comparison/` containing:

| Output | Description |
|--------|-------------|
| `cross_model_report.md` | Full comparison report with tables and findings |
| `combined_effects.csv` | Per-trait effect sizes across all models |
| `combined_judge_qwk.csv` | Judge QWK scores across all models |
| `plots/` | 7 comparative visualizations |

**Key cross-model findings (Qwen3-4B vs Qwen3-32B vs gpt-oss-20b):**

| Metric | Qwen3-4B | Qwen3-32B | gpt-oss-20b |
|--------|----------|-----------|-------------|
| Mean vulnerability (\|pos effect\|) | 0.172 | 0.021 | 0.123 |
| Most destructive trait | humorous (-0.42) | hallucinating (-0.06) | evil (-0.44) |
| Unsteered judge QWK | 0.184 | 0.253 | 0.239 |
| OpenAI judge QWK | 0.500 | 0.518 | 0.468 |

- **Smaller models are more vulnerable**: Qwen3-4B is 8× more susceptible to steering than Qwen3-32B
- **The most destructive trait varies by model**: humorous (4B), hallucinating (32B), evil (20B)
- **Qwen3-32B is the best unsteered judge** but all local models lag far behind OpenAI

### Pre-Built Pipeline Script

For convenience, `run_gpt_oss_pipeline.sh` runs the entire pipeline end-to-end for gpt-oss-20b. Adapt it for other models by changing the `MODEL`, `SHORT_NAME`, and batch size variables.

### Results

Results are saved to `experiments/education/results/multi_trait_<model>_<timestamp>/`:

| File/Dir | Description |
|----------|-------------|
| `config.json` | Experiment configuration and parameters |
| `experiment_a_answers.jsonl` | Student-generated answers (1,500 per run) |
| `experiment_a_llm_scores.jsonl` | LLM judge scores for student answers |
| `experiment_a_openai_scores.jsonl` | OpenAI judge scores for student answers |
| `experiment_b_judge_scores.jsonl` | Judge scores on real ASAP-SAS essays (1,600 per run) |
| `shared/sampled_essays.jsonl` | Sampled essays for Experiment B |
| `report/` | Analysis report with plots |

### Essay Set Types

| Sets | Type | Topics |
|------|------|--------|
| 1, 10 | Science Experiment | Acid Rain, Heat Absorption |
| 3, 4 | Reading Comprehension | Invasive Species article |
| 5, 6 | Science Knowledge | Protein Synthesis, Cell Membrane |
| 7, 8 | Literary Analysis | "Crossing Over", "Gifts" stories |
| 9 | Informational Text | Space Junk article organization |
| 2 | Opinion/Discussion | Library Censorship |

### Dataset

The ASAP-SAS DOCX files should be in `asap-sas/` directory:
- `Data Set #1--ReadMeFirst.docx` through `Data Set #10--ReadMeFirst.docx`
- These contain the prompts, reading passages, and scoring rubrics

The `essay_configs.json` file is pre-generated and included in the repo.

### Simple Experiment (Legacy)

A simpler 3×3 experiment (good/evil/unsteered × 3 judge types) is also available:

```bash
python -m experiments.education.run_simple_experiment \
    --model Qwen/Qwen3-4B \
    --vector-path persona_vectors/Qwen3-4B/evil_response_avg_diff.pt \
    --layer 20 --coef 2.0 --samples 5
```
