#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-llama31_8b}"
REWRITER_MODEL="${REWRITER_MODEL:-qwen25_7b}"
REWARD_MODEL="${REWARD_MODEL:-skywork_reward}"
CACHE_DIR="${CACHE_DIR:-models}"
RUN_DIR="${RUN_DIR:-runs/llama31_8b/mtbench}"
PYTHON="${PYTHON:-python}"
TP="${TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BACKEND="${BACKEND:-auto}"
SCORED_PATH="${SCORED_PATH:-$RUN_DIR/scored_context_hybrid_cleaned.jsonl}"

mkdir -p "$RUN_DIR"

"$PYTHON" -m darc.generate \
  --dataset mtbench \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --out "$RUN_DIR/candidates.jsonl" \
  --download \
  --k "${K:-16}" \
  --temperature "${GEN_TEMPERATURE:-0.8}" \
  --top-p "${GEN_TOP_P:-0.98}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-320}" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --backend "$BACKEND" \
  --batch-size "${GEN_BATCH_SIZE:-8}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-1}" \
  --seed "${GEN_SEED:-7}"

"$PYTHON" -m darc.perturb \
  --candidates "$RUN_DIR/candidates.jsonl" \
  --out "$RUN_DIR/perturbed_hybrid.jsonl" \
  --rewriter-model "$REWRITER_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --download \
  --naug "${NAUG:-8}" \
  --mode "${PERTURB_MODE:-hybrid}" \
  --temperature "${PERTURB_TEMPERATURE:-0.7}" \
  --top-p "${PERTURB_TOP_P:-0.95}" \
  --max-new-tokens "${PERTURB_MAX_NEW_TOKENS:-360}" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --backend "$BACKEND" \
  --batch-size "${PERTURB_BATCH_SIZE:-8}" \
  --sample-batch-size "${REWRITE_SAMPLE_BATCH_SIZE:-1}" \
  --rewrite-prompt-batch-size "${REWRITE_PROMPT_BATCH_SIZE:-8}" \
  --row-batch-size "${PERTURB_ROW_BATCH_SIZE:-1}" \
  --seed "${PERTURB_SEED:-17}"

"$PYTHON" -m darc.reward \
  --perturbed "$RUN_DIR/perturbed_hybrid.jsonl" \
  --out "$SCORED_PATH" \
  --reward-model "$REWARD_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --download \
  --batch-size "${REWARD_BATCH_SIZE:-16}"

"$PYTHON" -m darc.select \
  --scored "$SCORED_PATH" \
  --out "$RUN_DIR/selected_context_darc_eps_cleanbudget_target_met.jsonl" \
  --methods base darc_eps \
  --beta "${DARC_BETA:-0.75}" \
  --eps "${DARC_EPS:-1.5}" \
  --q-rp "${DARC_Q_RP:-0.25}" \
  --reward-objective clean \
  --reward-budget "${REWARD_BUDGET:-0.149}" \
  --budget-step "${BUDGET_STEP:-0.001}"

"$PYTHON" -m darc.evaluate \
  --selected "$RUN_DIR/selected_context_darc_eps_cleanbudget_target_met.jsonl" \
  --out-dir "$RUN_DIR/metrics_context_darc_eps_cleanbudget_target_met" \
  --methods base darc_eps \
  --lambda-risk "${LAMBDA_RISK:-1.99}" \
  --reward-field clean

"$PYTHON" -m darc.export_results \
  --selected "$RUN_DIR/selected_context_darc_eps_cleanbudget_target_met.jsonl" \
  --method darc_eps \
  --model-id "${MODEL_ID:-llama31_8b_darc_eps}" \
  --out "$RUN_DIR/mtbench_llama31_8b_darc_eps_answers.jsonl"
