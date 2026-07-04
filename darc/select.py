from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .io import read_jsonl, write_jsonl
from .math_utils import cvar_low, entropic_value, mean_std


def candidate_stats(scores: list[float], beta: float) -> dict:
    mu, sigma = mean_std(scores)
    v = entropic_value(scores, beta=beta)
    return {
        "mean": mu,
        "std": sigma,
        "entropic": v,
        "risk_premium": mu - v,
        "cvar10": cvar_low(scores, alpha=0.1),
        "n": len(scores),
    }


def choose(stats: list[dict], method: str, eps: float, q_rp: float) -> int:
    if method == "base":
        return int(np.argmax([s["mean"] for s in stats]))
    if method == "darc":
        return int(np.argmax([s["entropic"] for s in stats]))
    if method == "darc_tau":
        premiums = np.asarray([s["risk_premium"] for s in stats], dtype=np.float64)
        tau = float(np.quantile(premiums, q_rp))
        feasible = [i for i, s in enumerate(stats) if s["risk_premium"] <= tau]
        if not feasible:
            feasible = list(range(len(stats)))
        return int(max(feasible, key=lambda i: stats[i]["mean"]))
    if method == "darc_eps":
        robust_values = [s["entropic"] for s in stats]
        vmax = max(robust_values)
        feasible = [i for i, v in enumerate(robust_values) if v >= vmax - eps]
        return int(min(feasible, key=lambda i: (stats[i]["std"], -stats[i]["mean"])))
    raise ValueError(f"Unknown method: {method}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=["base", "darc", "darc_tau", "darc_eps"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--q-rp", type=float, default=0.25)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_rows = []
    for row in tqdm(list(read_jsonl(args.scored)), desc="select"):
        stats = [candidate_stats(c["scores"], beta=args.beta) for c in row["scored_candidates"]]
        selections = {}
        for method in args.methods:
            idx = choose(stats, method=method, eps=args.eps, q_rp=args.q_rp)
            candidate = row["scored_candidates"][idx]
            selections[method] = {
                "candidate_index": idx,
                "candidate_id": candidate["candidate_id"],
                "answers": candidate["answers"],
                "stats": stats[idx],
            }
        row["candidate_stats"] = stats
        row["selections"] = selections
        row["darc_params"] = {"beta": args.beta, "eps": args.eps, "q_rp": args.q_rp}
        out_rows.append(row)
    write_jsonl(Path(args.out), out_rows)


if __name__ == "__main__":
    main()
