from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .io import read_jsonl, write_jsonl
from .math_utils import cvar_low, entropic_value, mean_std


def clean_reward(candidate: dict, scores: list[float]) -> float:
    vals = []
    for turn in candidate.get("scored_turns", []):
        turn_scores = turn.get("scores", [])
        if turn_scores:
            vals.append(float(turn_scores[0]))
    if vals:
        return float(np.mean(vals))
    return mean_std(scores)[0]


def candidate_stats(candidate_or_scores: dict | list[float], beta: float) -> dict:
    if isinstance(candidate_or_scores, dict):
        candidate = candidate_or_scores
        scores = [float(x) for x in candidate["scores"]]
        clean = clean_reward(candidate, scores)
    else:
        scores = [float(x) for x in candidate_or_scores]
        clean = mean_std(scores)[0]
    mu, sigma = mean_std(scores)
    v = entropic_value(scores, beta=beta)
    return {
        "mean": mu,
        "clean_reward": clean,
        "std": sigma,
        "entropic": v,
        "risk_premium": mu - v,
        "cvar10": cvar_low(scores, alpha=0.1),
        "n": len(scores),
    }


def reward_value(stat: dict, objective: str) -> float:
    if objective == "clean":
        return float(stat.get("clean_reward", stat["mean"]))
    if objective == "mean":
        return float(stat["mean"])
    raise ValueError(f"Unknown reward objective: {objective}")


def choose(stats: list[dict], method: str, eps: float, q_rp: float, reward_objective: str = "clean") -> int:
    if method == "base":
        return int(np.argmax([reward_value(s, reward_objective) for s in stats]))
    if method == "darc":
        return int(np.argmax([s["entropic"] for s in stats]))
    if method == "darc_tau":
        premiums = np.asarray([s["risk_premium"] for s in stats], dtype=np.float64)
        tau = float(np.quantile(premiums, q_rp))
        feasible = [i for i, s in enumerate(stats) if s["risk_premium"] <= tau]
        if not feasible:
            feasible = list(range(len(stats)))
        return int(max(feasible, key=lambda i: reward_value(stats[i], reward_objective)))
    if method == "darc_eps":
        robust_values = [s["entropic"] for s in stats]
        vmax = max(robust_values)
        feasible = [i for i, v in enumerate(robust_values) if v >= vmax - eps]
        return int(min(feasible, key=lambda i: (stats[i]["std"], -reward_value(stats[i], reward_objective))))
    raise ValueError(f"Unknown method: {method}")


def darc_eps_candidates(stats: list[dict], eps: float, base_idx: int) -> list[int]:
    robust_values = [s["entropic"] for s in stats]
    vmax = max(robust_values)
    feasible = [i for i, v in enumerate(robust_values) if v >= vmax - eps]
    if base_idx not in feasible:
        feasible.append(base_idx)
    return feasible


def budgeted_darc_eps_indices(
    all_stats: list[list[dict]],
    eps: float,
    reward_objective: str,
    reward_budget: float,
    budget_step: float,
) -> list[int]:
    if reward_budget < 0:
        return [choose(stats, "darc_eps", eps=eps, q_rp=0.0, reward_objective=reward_objective) for stats in all_stats]
    if budget_step <= 0:
        raise ValueError("budget_step must be positive")

    n = len(all_stats)
    total_budget = int(round(reward_budget * n / budget_step))
    base_indices = [choose(stats, "base", eps=eps, q_rp=0.0, reward_objective=reward_objective) for stats in all_stats]
    option_sets = []
    for stats, base_idx in zip(all_stats, base_indices):
        base_reward = reward_value(stats[base_idx], reward_objective)
        base_risk = float(stats[base_idx]["std"])
        best_by_cost: dict[int, tuple[float, int]] = {0: (0.0, base_idx)}
        for idx in darc_eps_candidates(stats, eps=eps, base_idx=base_idx):
            cost = max(0.0, base_reward - reward_value(stats[idx], reward_objective))
            cost_units = int(round(cost / budget_step))
            if cost_units > total_budget:
                continue
            gain = base_risk - float(stats[idx]["std"])
            old = best_by_cost.get(cost_units)
            if old is None or (gain, reward_value(stats[idx], reward_objective)) > (
                old[0],
                reward_value(stats[old[1]], reward_objective),
            ):
                best_by_cost[cost_units] = (gain, idx)
        option_sets.append([(cost, gain_idx[0], gain_idx[1]) for cost, gain_idx in best_by_cost.items()])

    neg = -1.0e30
    dp = np.full((n + 1, total_budget + 1), neg, dtype=np.float64)
    back_cost = np.full((n + 1, total_budget + 1), -1, dtype=np.int32)
    back_idx = np.full((n + 1, total_budget + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for row_idx, options in enumerate(option_sets, start=1):
        prev = dp[row_idx - 1]
        cur = dp[row_idx]
        for used in np.flatnonzero(prev > neg / 2):
            prev_score = prev[used]
            for cost, gain, cand_idx in options:
                new_used = int(used + cost)
                if new_used > total_budget:
                    continue
                score = prev_score + gain
                if score > cur[new_used]:
                    cur[new_used] = score
                    back_cost[row_idx, new_used] = cost
                    back_idx[row_idx, new_used] = cand_idx

    used = int(np.argmax(dp[n]))
    selected = [0] * n
    for row_idx in range(n, 0, -1):
        idx = int(back_idx[row_idx, used])
        if idx < 0:
            idx = base_indices[row_idx - 1]
        selected[row_idx - 1] = idx
        cost = int(back_cost[row_idx, used])
        if cost > 0:
            used -= cost
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=["base", "darc", "darc_tau", "darc_eps"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--q-rp", type=float, default=0.25)
    p.add_argument("--reward-objective", choices=["clean", "mean"], default="clean")
    p.add_argument("--reward-budget", type=float, default=0.149)
    p.add_argument("--budget-step", type=float, default=0.001)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = list(read_jsonl(args.scored))
    stats_by_row = [
        [candidate_stats(candidate, beta=args.beta) for candidate in row["scored_candidates"]]
        for row in tqdm(rows, desc="score candidates")
    ]
    budgeted_eps = None
    if "darc_eps" in args.methods:
        budgeted_eps = budgeted_darc_eps_indices(
            stats_by_row,
            eps=args.eps,
            reward_objective=args.reward_objective,
            reward_budget=args.reward_budget,
            budget_step=args.budget_step,
        )

    out_rows = []
    for row_idx, row in enumerate(tqdm(rows, desc="select")):
        stats = stats_by_row[row_idx]
        selections = {}
        for method in args.methods:
            if method == "darc_eps" and budgeted_eps is not None:
                idx = budgeted_eps[row_idx]
            else:
                idx = choose(
                    stats,
                    method=method,
                    eps=args.eps,
                    q_rp=args.q_rp,
                    reward_objective=args.reward_objective,
                )
            candidate = row["scored_candidates"][idx]
            selections[method] = {
                "candidate_index": idx,
                "candidate_id": candidate["candidate_id"],
                "answers": candidate["answers"],
                "stats": stats[idx],
            }
        row["candidate_stats"] = stats
        row["selections"] = selections
        row["darc_params"] = {
            "beta": args.beta,
            "eps": args.eps,
            "q_rp": args.q_rp,
            "reward_objective": args.reward_objective,
            "reward_budget": args.reward_budget,
            "budget_step": args.budget_step,
        }
        out_rows.append(row)
    write_jsonl(Path(args.out), out_rows)


if __name__ == "__main__":
    main()
