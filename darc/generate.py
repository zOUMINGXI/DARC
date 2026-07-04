from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from .data import load_prompts
from .io import append_jsonl_row, read_jsonl
from .modelscope_utils import resolve_model
from .templates import messages_for_turns, prompt_text
from .textgen import GenerationConfig, make_generator, pick_backend


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mtbench", "alpacaeval2"])
    p.add_argument("--model", required=True, help="ModelScope id, HF id, local path, or alias.")
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--prompt-batch-size",
        type=int,
        default=1,
        help="Number of benchmark prompts to advance together. Default preserves one-prompt-at-a-time behavior.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--download", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    existing_ids = set()
    if out_path.exists():
        existing_ids = {str(row["prompt_id"]) for row in read_jsonl(out_path)}
    prompts = [p for p in load_prompts(args.dataset, limit=args.limit) if str(p["prompt_id"]) not in existing_ids]
    if not prompts:
        print(f"All prompts already generated in {out_path}")
        return

    model_path = resolve_model(args.model, cache_dir=args.cache_dir, download=args.download)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    backend = pick_backend(args.backend)
    generator = make_generator(
        model_path=model_path,
        backend=backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = GenerationConfig(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    one_sampling = GenerationConfig(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    prompt_batch_size = max(1, args.prompt_batch_size)
    with tqdm(total=len(prompts), desc=f"generate {args.dataset}") as pbar:
        for prompt_group in batched(prompts, prompt_batch_size):
            group_answers: list[list[list[str]]] = [[[] for _ in range(args.k)] for _ in prompt_group]
            max_turns = max(len(row["turns"]) for row in prompt_group)
            for turn_idx in range(max_turns):
                active = [i for i, row in enumerate(prompt_group) if turn_idx < len(row["turns"])]
                if turn_idx == 0:
                    texts = []
                    slots = []
                    for group_idx in active:
                        current_turns = prompt_group[group_idx]["turns"][:1]
                        texts.append(prompt_text(tokenizer, messages_for_turns(current_turns, [])))
                        slots.append(group_idx)
                    outputs = generator.generate(texts, sampling, batch_size=args.batch_size)
                    for group_idx, out in zip(slots, outputs):
                        if len(out) != args.k:
                            raise RuntimeError(f"Expected {args.k} samples, got {len(out)}")
                        for candidate_idx, output in enumerate(out):
                            group_answers[group_idx][candidate_idx].append(output.strip())
                else:
                    texts = []
                    slots = []
                    for group_idx in active:
                        current_turns = prompt_group[group_idx]["turns"][: turn_idx + 1]
                        for candidate_idx in range(args.k):
                            texts.append(
                                prompt_text(
                                    tokenizer,
                                    messages_for_turns(current_turns, group_answers[group_idx][candidate_idx]),
                                )
                            )
                            slots.append((group_idx, candidate_idx))
                    outputs = generator.generate(texts, one_sampling, batch_size=args.batch_size)
                    for (group_idx, candidate_idx), out in zip(slots, outputs):
                        if not out:
                            raise RuntimeError("Generator returned no samples")
                        group_answers[group_idx][candidate_idx].append(out[0].strip())

            for prompt_row, candidate_answers in zip(prompt_group, group_answers):
                candidates = [{"candidate_id": k, "answers": candidate_answers[k]} for k in range(args.k)]
                append_jsonl_row(
                    out_path,
                    {
                        "dataset": args.dataset,
                        "prompt_id": prompt_row["prompt_id"],
                        "turns": prompt_row["turns"],
                        "candidates": candidates,
                        "raw": prompt_row.get("raw", {}),
                        "model": str(model_path),
                        "generation": {
                            "k": args.k,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "seed": args.seed,
                            "backend": backend,
                            "prompt_batch_size": prompt_batch_size,
                        },
                    }
                )
                pbar.update(1)


if __name__ == "__main__":
    main()
