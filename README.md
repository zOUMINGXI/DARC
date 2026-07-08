# DARC-eps

Source code for DARC-eps, a risk-aware decoding and selection pipeline for instruction-tuned language models.

DARC-eps scores multiple candidate answers under perturbations, computes an entropic robust value for each candidate, keeps candidates inside an epsilon robust-value set, and selects lower-risk candidates under a clean reward budget.

## Install

```bash
pip install -e .
```

## Pipeline

Generate candidate answers:

```bash
darc-generate \
  --dataset mtbench \
  --model <model_path_or_id> \
  --out runs/mtbench/candidates.jsonl
```

Build perturbations:

```bash
darc-perturb \
  --candidates runs/mtbench/candidates.jsonl \
  --out runs/mtbench/perturbed.jsonl
```

Score original and perturbed answers:

```bash
darc-reward \
  --reward-model <reward_model_path_or_id> \
  --perturbed runs/mtbench/perturbed.jsonl \
  --out runs/mtbench/scored.jsonl
```

Select with DARC-eps:

```bash
darc-select \
  --scored runs/mtbench/scored.jsonl \
  --out runs/mtbench/selected.jsonl \
  --methods base darc_eps \
  --beta 1.0 \
  --eps 0.25 \
  --reward-objective clean \
  --reward-budget 0.149 \
  --budget-step 0.001
```

Evaluate proxy reward-risk metrics:

```bash
darc-eval \
  --selected runs/mtbench/selected.jsonl \
  --out-dir runs/mtbench/metrics \
  --methods base darc_eps \
  --reward-field clean
```

Export selected answers for MT-Bench style evaluation:

```bash
darc-export \
  --selected runs/mtbench/selected.jsonl \
  --method darc_eps \
  --model-id qwen25_7b_darc_eps \
  --out runs/mtbench/answers.jsonl
```

## Qwen2.5-7B MT-Bench

The checked recipe uses the DARC-eps clean-budget selection used for the Qwen2.5-7B-Instruct MT-Bench reproduction:

```bash
bash recipes/qwen25_7b_mtbench_darc_eps.sh
```

Default selection parameters:

```text
beta=0.75
eps=1.5
reward_budget=0.149
budget_step=0.001
reward_objective=clean
```

The script writes generated candidates, perturbations, reward scores, selected answers, proxy metrics, and MT-Bench-format answers under `runs/qwen25_7b/mtbench`. These artifacts are intentionally ignored by git.

## Selection Rule

For each candidate, DARC-eps computes:

```text
V_beta(a) = -1 / beta * log mean_i exp(-beta * r_i(a))
```

The method keeps candidates whose robust value is within `eps` of the best robust value for the prompt and uses the base candidate as a zero-drop fallback in the budget solver. It then solves a global clean-reward-budgeted selection problem that maximizes risk reduction while keeping the average clean reward drop under the configured budget.
