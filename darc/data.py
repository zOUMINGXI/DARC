from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from datasets import load_dataset

FASTCHAT_MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def load_alpacaeval(limit: int | None = None) -> list[dict]:
    local_path = Path("data") / "alpaca_eval" / "alpaca_eval.json"
    if local_path.exists():
        ds = json.loads(local_path.read_text(encoding="utf-8"))
    else:
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    rows: list[dict] = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(
            {
                "dataset": "alpacaeval2",
                "prompt_id": str(item.get("instruction_id", i)),
                "turns": [item["instruction"]],
                "raw": dict(item),
            }
        )
    return rows


def load_mtbench(fastchat_dir: str | Path | None = None, limit: int | None = None) -> list[dict]:
    if fastchat_dir is not None:
        path = Path(fastchat_dir) / "fastchat" / "llm_judge" / "data" / "mt_bench" / "question.jsonl"
        questions = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        local_path = Path("data") / "mt_bench" / "question.jsonl"
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as f:
                questions = [json.loads(line) for line in f if line.strip()]
        else:
            try:
                from fastchat.llm_judge.common import load_questions
                import fastchat

                pkg_root = Path(fastchat.__file__).resolve().parent
                path = pkg_root / "llm_judge" / "data" / "mt_bench" / "question.jsonl"
                questions = load_questions(str(path), None, None)
            except Exception:
                with urllib.request.urlopen(FASTCHAT_MT_BENCH_URL, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
                questions = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows: list[dict] = []
    for i, q in enumerate(questions):
        if limit is not None and i >= limit:
            break
        rows.append(
            {
                "dataset": "mtbench",
                "prompt_id": str(q.get("question_id", i)),
                "category": q.get("category"),
                "turns": list(q["turns"]),
                "raw": q,
            }
        )
    return rows


def load_prompts(dataset: str, limit: int | None = None) -> list[dict]:
    name = dataset.lower().replace("_", "").replace("-", "")
    if name in {"alpacaeval", "alpacaeval2", "alpacaeval20"}:
        return load_alpacaeval(limit=limit)
    if name in {"mtbench", "mt"}:
        return load_mtbench(limit=limit)
    raise ValueError(f"Unknown dataset: {dataset}")
