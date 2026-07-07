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
  --model <model_path_or_id> \
  --out runs/mtbench/candidates.jsonl
```

Build perturbations:

```bash
darc-perturb \
  --input runs/mtbench/candidates.jsonl \
  --out runs/mtbench/perturbed.jsonl
```

Score original and perturbed answers:

```bash
darc-reward \
  --model <reward_model_path_or_id> \
  --input runs/mtbench/perturbed.jsonl \
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
  --out runs/mtbench/answers.jsonl
```

## Selection Rule

For each candidate, DARC-eps computes:

```text
V_beta(a) = -1 / beta * log mean_i exp(-beta * r_i(a))
```

The method keeps candidates whose robust value is within `eps` of the best robust value for the prompt and uses the base candidate as a zero-drop fallback in the budget solver. It then solves a global clean-reward-budgeted selection problem that maximizes risk reduction while keeping the average clean reward drop under the configured budget.
