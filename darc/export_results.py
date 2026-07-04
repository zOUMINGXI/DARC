from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from .io import read_jsonl, write_jsonl


def export_alpacaeval(rows: list[dict], method: str, model_id: str, out: Path) -> None:
    payload = []
    for row in rows:
        answer = row["selections"][method]["answers"][0]
        payload.append(
            {
                "instruction": row["turns"][0],
                "output": answer,
                "generator": model_id,
                "dataset": "alpaca_eval",
            }
        )
    write_jsonl(out, payload)


def export_mtbench(rows: list[dict], method: str, model_id: str, out: Path) -> None:
    payload = []
    now = time.time()
    for row in rows:
        raw = row.get("raw", {})
        answers = row["selections"][method]["answers"]
        payload.append(
            {
                "question_id": raw.get("question_id", row["prompt_id"]),
                "answer_id": str(uuid.uuid4()),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": answers}],
                "tstamp": now,
            }
        )
    write_jsonl(out, payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True)
    p.add_argument("--method", default="darc_eps")
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = list(read_jsonl(args.selected))
    dataset = rows[0]["dataset"].lower()
    out = Path(args.out)
    if dataset == "alpacaeval2":
        export_alpacaeval(rows, args.method, args.model_id, out)
    elif dataset == "mtbench":
        export_mtbench(rows, args.method, args.model_id, out)
    else:
        raise ValueError(f"Unknown dataset in selected file: {dataset}")


if __name__ == "__main__":
    main()
