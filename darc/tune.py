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
from .select import candidate_stats, choose


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stable_is_calib(prompt_id: str, frac: float, seed: int) -> bool:
    key = f"{seed}:{prompt_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    val = int(h[:12], 16) / float(16**12)
    return val < frac


def apply_selection(rows: list[dict], beta: float, eps: float, q_rp: float, methods: list[str]) -> list[dict]:
    selected = []
    for row in rows:
        stats = [candidate_stats(c["scores"], beta=beta) for c in row["scored_candidates"]]
        selections = {}
        for method in methods:
            idx = choose(stats, method=method, eps=eps, q_rp=q_rp)
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
        new_row["darc_params"] = {"beta": beta, "eps": eps, "q_rp": q_rp}
        selected.append(new_row)
    return selected


def score_selected(rows: list[dict], method: str, lam: float) -> float:
    vals = []
    for row in rows:
        st = row["selections"][method]["stats"]
        vals.append(st["mean"] - lam * st["std"])
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
    for beta, eps, q_rp in tqdm(
        list(itertools.product(parse_floats(args.betas), parse_floats(args.eps_values), parse_floats(args.q_rp_values))),
        desc="tune grid",
    ):
        methods = ["base", args.method]
        selected = apply_selection(calib_rows, beta=beta, eps=eps, q_rp=q_rp, methods=methods)
        records.append(
            {
                "beta": beta,
                "eps": eps,
                "q_rp": q_rp,
                "method": args.method,
                "calib_n": len(calib_rows),
                "calib_tradeoff": score_selected(selected, args.method, lam=args.lambda_risk),
                "base_tradeoff": score_selected(selected, "base", lam=args.lambda_risk),
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
            methods=["base", args.method],
        )
        write_jsonl(args.selected_out, selected_all)


if __name__ == "__main__":
    main()
