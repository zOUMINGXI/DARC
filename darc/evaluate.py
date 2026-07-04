from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import ensure_dir, read_jsonl


def summarize(rows: list[dict], method: str, lam: float, hv_top_p: float) -> list[dict]:
    per_prompt = []
    for row in rows:
        sel = row["selections"][method]
        st = sel["stats"]
        base_std = row["selections"]["base"]["stats"]["std"]
        per_prompt.append(
            {
                "dataset": row["dataset"],
                "prompt_id": row["prompt_id"],
                "method": method,
                "reward": st["mean"],
                "risk": st["std"],
                "tradeoff": st["mean"] - lam * st["std"],
                "cvar10": st["cvar10"],
                "len_tok_proxy": sum(len(a.split()) for a in sel["answers"]),
                "base_risk": base_std,
            }
        )
    full = aggregate(per_prompt, "overall")
    n_hv = max(1, int(np.ceil(len(per_prompt) * hv_top_p)))
    hv_rows = sorted(per_prompt, key=lambda x: x["base_risk"], reverse=True)[:n_hv]
    hv = aggregate(hv_rows, f"high_variance_top_{int(hv_top_p * 100)}")
    return [full, hv]


def aggregate(rows: list[dict], subset: str) -> dict:
    arr = pd.DataFrame(rows)
    return {
        "subset": subset,
        "method": rows[0]["method"],
        "n": int(len(rows)),
        "reward": float(arr["reward"].mean()),
        "risk": float(arr["risk"].mean()),
        "tradeoff": float(arr["tradeoff"].mean()),
        "cvar10_prompt": float(np.mean(np.sort(arr["reward"].to_numpy())[: max(1, int(np.ceil(0.1 * len(arr))))])),
        "len_tok_proxy": float(arr["len_tok_proxy"].mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", nargs="+", default=["base", "darc", "darc_tau", "darc_eps"])
    p.add_argument("--lambda-risk", type=float, default=1.99)
    p.add_argument("--hv-top-p", type=float, default=0.20)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = list(read_jsonl(args.selected))
    out_dir = ensure_dir(args.out_dir)
    summaries = []
    for method in args.methods:
        summaries.extend(summarize(rows, method=method, lam=args.lambda_risk, hv_top_p=args.hv_top_p))
    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "proxy_metrics.csv", index=False)
    with (out_dir / "proxy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
