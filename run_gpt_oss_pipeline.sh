#!/bin/bash
# Pipeline for gpt-oss-20b: generate persona vectors + run education experiment
# Model: openai/gpt-oss-20b (24 layers, hidden_size=2880, MoE)
# Steering layer: 12 (num_layers // 2)

set -e
cd /root/persona_vectors

export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export $(cat .env | xargs)

MODEL="openai/gpt-oss-20b"
SHORT_NAME="gpt-oss-20b"
TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic")
VECTOR_DIR="persona_vectors/${SHORT_NAME}"
EXTRACT_DIR="eval_persona_extract"

mkdir -p "$VECTOR_DIR"
mkdir -p "$EXTRACT_DIR"

echo "========================================="
echo "Pipeline for $MODEL"
echo "========================================="

# ---- Step 1: Generate positive persona responses ----
echo ""
echo "===== STEP 1: Generating POSITIVE persona responses ====="
for trait in "${TRAITS[@]}"; do
    POS_PATH="${EXTRACT_DIR}/${SHORT_NAME}_${trait}_pos.csv"
    if [ -f "$POS_PATH" ]; then
        echo "  [SKIP] $POS_PATH already exists"
        continue
    fi
    echo "  [RUN] Generating positive responses for trait: $trait"
    python -m eval.eval_persona \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$POS_PATH" \
        --coef 0 \
        --persona_instruction_type pos \
        --n_per_question 10 \
        --max_tokens 1000 \
        --batch_process True \
        --max_concurrent_judges 100
    echo "  [DONE] $trait positive"
done

# ---- Step 2: Generate negative persona responses ----
echo ""
echo "===== STEP 2: Generating NEGATIVE persona responses ====="
for trait in "${TRAITS[@]}"; do
    NEG_PATH="${EXTRACT_DIR}/${SHORT_NAME}_${trait}_neg.csv"
    if [ -f "$NEG_PATH" ]; then
        echo "  [SKIP] $NEG_PATH already exists"
        continue
    fi
    echo "  [RUN] Generating negative responses for trait: $trait"
    python -m eval.eval_persona \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$NEG_PATH" \
        --coef 0 \
        --persona_instruction_type neg \
        --n_per_question 10 \
        --max_tokens 1000 \
        --batch_process True \
        --max_concurrent_judges 100
    echo "  [DONE] $trait negative"
done

# ---- Step 3: Generate steering vectors ----
echo ""
echo "===== STEP 3: Generating steering vectors ====="
for trait in "${TRAITS[@]}"; do
    POS_PATH="${EXTRACT_DIR}/${SHORT_NAME}_${trait}_pos.csv"
    NEG_PATH="${EXTRACT_DIR}/${SHORT_NAME}_${trait}_neg.csv"
    VEC_FILE="${VECTOR_DIR}/${trait}_prompt_avg_diff.pt"
    if [ -f "$VEC_FILE" ]; then
        echo "  [SKIP] Vectors for $trait already exist"
        continue
    fi
    echo "  [RUN] Generating vectors for trait: $trait"
    python generate_vec.py \
        --model_name "$MODEL" \
        --pos_path "$POS_PATH" \
        --neg_path "$NEG_PATH" \
        --trait "$trait" \
        --save_dir "$VECTOR_DIR" \
        --threshold 50
    echo "  [DONE] $trait vectors"
done

echo ""
echo "===== Vectors generated. Listing: ====="
ls -la "$VECTOR_DIR/"

# ---- Step 4: Run education experiment ----
echo ""
echo "===== STEP 4: Running education experiment ====="
python -m experiments.education.run_multi_trait_experiment \
    --model "$MODEL" \
    --layer 12 \
    --gen-batch-size 4 \
    --score-batch-size 8

echo ""
echo "===== STEP 5: Generate analysis report ====="
# Find the latest results directory for this model
RESULTS_DIR=$(ls -td experiments/education/results/multi_trait_${SHORT_NAME}_* 2>/dev/null | head -1)
if [ -n "$RESULTS_DIR" ]; then
    python experiments/education/generate_report.py --results-dir "$RESULTS_DIR"
    echo "Report generated in $RESULTS_DIR/report/"
else
    echo "WARNING: Could not find results directory for $SHORT_NAME"
fi

echo ""
echo "========================================="
echo "Pipeline complete for $MODEL"
echo "========================================="
