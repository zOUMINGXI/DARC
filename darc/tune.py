from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io import dump_json, read_jsonl, write_jsonl
from .select import budgeted_darc_eps_indices, candidate_stats, choose


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stable_is_calib(prompt_id: str, frac: float, seed: int) -> bool:
    key = f"{seed}:{prompt_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    val = int(h[:12], 16) / float(16**12)
    return val < frac


def apply_selection(
    rows: list[dict],
    beta: float,
    eps: float,
    q_rp: float,
    methods: list[str],
    reward_objective: str,
    reward_budget: float,
    budget_step: float,
) -> list[dict]:
    selected = []
    stats_by_row = [[candidate_stats(candidate, beta=beta) for candidate in row["scored_candidates"]] for row in rows]
    budgeted_eps = None
    if "darc_eps" in methods:
        budgeted_eps = budgeted_darc_eps_indices(
            stats_by_row,
            eps=eps,
            reward_objective=reward_objective,
            reward_budget=reward_budget,
            budget_step=budget_step,
        )
    for row_idx, row in enumerate(rows):
        stats = stats_by_row[row_idx]
        selections = {}
        for method in methods:
            if method == "darc_eps" and budgeted_eps is not None:
                idx = budgeted_eps[row_idx]
            else:
                idx = choose(stats, method=method, eps=eps, q_rp=q_rp, reward_objective=reward_objective)
            candidate = row["scored_candidates"][idx]
            selections[method] = {
                "candidate_index": idx,
                "candidate_id": candidate["candidate_id"],
                "answers": candidate["answers"],
                "stats": stats[idx],
            }
        new_row = dict(row)
        new_row["candidate_stats"] = stats
        new_row["selections"] = selections
        new_row["darc_params"] = {
            "beta": beta,
            "eps": eps,
            "q_rp": q_rp,
            "reward_objective": reward_objective,
            "reward_budget": reward_budget,
            "budget_step": budget_step,
        }
        selected.append(new_row)
    return selected


def stat_reward(stat: dict, reward_objective: str) -> float:
    if reward_objective == "clean":
        return float(stat.get("clean_reward", stat["mean"]))
    if reward_objective == "mean":
        return float(stat["mean"])
    raise ValueError(f"Unknown reward objective: {reward_objective}")


def score_selected(rows: list[dict], method: str, lam: float, reward_objective: str) -> float:
    vals = []
    for row in rows:
        st = row["selections"][method]["stats"]
        vals.append(stat_reward(st, reward_objective) - lam * st["std"])
    return float(np.mean(vals))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--selected-out", default=None)
    p.add_argument("--method", default="darc_eps", choices=["darc", "darc_tau", "darc_eps"])
    p.add_argument("--betas", default="0.25,0.5,1.0,2.0,3.0")
    p.add_argument("--eps-values", default="0.0,0.1,0.2,0.25,0.3,0.5")
    p.add_argument("--q-rp-values", default="0.1,0.25,0.5")
    p.add_argument("--reward-budget-values", default="0.05,0.1,0.149,0.2")
    p.add_argument("--reward-objective", choices=["clean", "mean"], default="clean")
    p.add_argument("--budget-step", type=float, default=0.001)
    p.add_argument("--lambda-risk", type=float, default=1.99)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = list(read_jsonl(args.scored))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_rows = [r for r in rows if stable_is_calib(str(r["prompt_id"]), args.calib_frac, args.seed)]
    if not calib_rows:
        calib_rows = rows[: max(1, int(len(rows) * args.calib_frac))]

    records = []
    budget_values = parse_floats(args.reward_budget_values) if args.method == "darc_eps" else [-1.0]
    for beta, eps, q_rp, reward_budget in tqdm(
        list(
            itertools.product(
                parse_floats(args.betas),
                parse_floats(args.eps_values),
                parse_floats(args.q_rp_values),
                budget_values,
            )
        ),
        desc="tune grid",
    ):
        methods = ["base", args.method]
        selected = apply_selection(
            calib_rows,
            beta=beta,
            eps=eps,
            q_rp=q_rp,
            methods=methods,
            reward_objective=args.reward_objective,
            reward_budget=reward_budget,
            budget_step=args.budget_step,
        )
        records.append(
            {
                "beta": beta,
                "eps": eps,
                "q_rp": q_rp,
                "reward_budget": reward_budget,
                "method": args.method,
                "calib_n": len(calib_rows),
                "calib_tradeoff": score_selected(
                    selected, args.method, lam=args.lambda_risk, reward_objective=args.reward_objective
                ),
                "base_tradeoff": score_selected(
                    selected, "base", lam=args.lambda_risk, reward_objective=args.reward_objective
                ),
            }
        )
    df = pd.DataFrame(records).sort_values(["calib_tradeoff", "beta"], ascending=[False, True])
    df.to_csv(out_dir / "tune_grid.csv", index=False)
    best = df.iloc[0].to_dict()
    best["lambda_risk"] = args.lambda_risk
    best["calib_frac"] = args.calib_frac
    dump_json(out_dir / "best_config.json", best)
    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.selected_out:
        selected_all = apply_selection(
            rows,
            beta=float(best["beta"]),
            eps=float(best["eps"]),
            q_rp=float(best["q_rp"]),
            reward_objective=args.reward_objective,
            reward_budget=float(best["reward_budget"]),
            budget_step=args.budget_step,
            methods=["base", args.method],
        )
        write_jsonl(args.selected_out, selected_all)


if __name__ == "__main__":
    main()
