from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from .io import append_jsonl_row, read_jsonl
from .modelscope_utils import resolve_model
from .textgen import GenerationConfig, make_generator, pick_backend

STYLE_INSTRUCTION = (
    "Given the prompt and a candidate response, rewrite the response while preserving its meaning. "
    "Only change wording, phrasing, or formatting. Do not add, remove, or alter facts; do not change "
    "any numbers, dates, named entities, URLs, or citations. Keep the tone and overall style similar. "
    "Keep the length within +/-10%. Output only the rewritten response (no preamble)."
)

TARGETED_INSTRUCTION = (
    "Given the prompt and a candidate response, rewrite the response while preserving the task-relevant "
    "meaning and factual content. Normalize superficial reward-model artifacts: avoid excessive verbosity, "
    "remove unnecessary flattery or sycophantic/apologetic phrasing, simplify overly rigid formatting, and "
    "keep the answer concise. Do not change factual claims, numbers, dates, named entities, URLs, citations, "
    "or the final answer. Output only the rewritten response."
)


def numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:[\.,:/-]\d+)*%?", text)


def valid_rewrite(original: str, rewritten: str, strict_numbers: bool = True) -> bool:
    rewritten = rewritten.strip()
    if not rewritten or rewritten == original.strip():
        return False
    if len(rewritten) < 0.5 * max(1, len(original)) or len(rewritten) > 1.5 * max(1, len(original)):
        return False
    if strict_numbers and sorted(numbers(original)) != sorted(numbers(rewritten)):
        return False
    return True


def consecutive_mode_quota(modes: list[str], start: int) -> int:
    if start >= len(modes):
        return 0
    mode = modes[start]
    quota = 0
    for item in modes[start:]:
        if item != mode:
            break
        quota += 1
    return quota


def modes_for(naug: int, mode: str) -> list[str]:
    if mode == "targeted":
        return ["targeted"] * naug
    if mode == "hybrid":
        return ["style"] * (naug // 2) + ["targeted"] * (naug - naug // 2)
    return ["style"] * naug


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def rewrite_prompt(tokenizer, prompt: str, response: str, mode: str) -> str:
    instruction = TARGETED_INSTRUCTION if mode == "targeted" else STYLE_INSTRUCTION
    messages = [
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Candidate response:\n{response}\n\n"
                "Rewrite:"
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rewriter-model", required=True, help="Use the generator model by default.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--naug", type=int, default=8)
    p.add_argument("--mode", choices=["style", "targeted", "hybrid"], default="style")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=360)
    p.add_argument("--max-attempt-multiplier", type=int, default=4)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Number of rewrite samples requested per response attempt. Default preserves sequential attempts.",
    )
    p.add_argument(
        "--rewrite-prompt-batch-size",
        type=int,
        default=8,
        help="Number of candidate rewrite prompts to submit together.",
    )
    p.add_argument(
        "--row-batch-size",
        type=int,
        default=1,
        help="Number of benchmark rows to perturb together. Rows are still appended atomically after each group.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--download", action="store_true")
    p.add_argument("--no-strict-numbers", action="store_true")
    return p


def prepare_row_states(row: dict, args) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    states = []
    lookup = {}
    for candidate_idx, candidate_item in enumerate(row["candidates"]):
        for turn_idx, candidate in enumerate(candidate_item["answers"]):
            state = {
                "candidate_idx": candidate_idx,
                "turn_idx": turn_idx,
                "prompt": row["turns"][turn_idx],
                "candidate": candidate,
                "modes": modes_for(args.naug, args.mode),
                "variants": [candidate],
                "attempts": 0,
                "target_idx": 0,
            }
            lookup[(candidate_idx, turn_idx)] = state
            states.append(state)
    return states, lookup


def finalize_row(
    row: dict,
    lookup: dict[tuple[int, int], dict],
    args,
    model_path: Path,
    backend: str,
    sample_batch_size: int,
    rewrite_prompt_batch_size: int,
    row_batch_size: int,
) -> dict:
    perturbed_candidates = []
    for candidate_idx, candidate_item in enumerate(row["candidates"]):
        turn_variants = []
        for turn_idx, candidate in enumerate(candidate_item["answers"]):
            state = lookup[(candidate_idx, turn_idx)]
            variants = state["variants"]
            turn_variants.append(
                {
                    "response": candidate,
                    "variants": variants,
                    "accepted_aug": len(variants) - 1,
                }
            )
        perturbed_candidates.append(
            {
                "candidate_id": candidate_item["candidate_id"],
                "answers": candidate_item["answers"],
                "turn_variants": turn_variants,
            }
        )
    row["perturbed_candidates"] = perturbed_candidates
    row["perturbation"] = {
        "naug": args.naug,
        "mode": args.mode,
        "seed": args.seed,
        "rewriter_model": str(model_path),
        "backend": backend,
        "sample_batch_size": sample_batch_size,
        "rewrite_prompt_batch_size": rewrite_prompt_batch_size,
        "row_batch_size": row_batch_size,
        "max_num_seqs": args.max_num_seqs,
    }
    return row


def perturb_row_group(
    rows: list[dict],
    args,
    tokenizer,
    generator,
    model_path: Path,
    backend: str,
    sample_batch_size: int,
    rewrite_prompt_batch_size: int,
    row_batch_size: int,
) -> list[dict]:
    contexts = []
    states = []
    for row in rows:
        row_states, lookup = prepare_row_states(row, args)
        contexts.append((row, lookup))
        states.extend(row_states)

    max_attempts = args.naug * args.max_attempt_multiplier
    round_idx = 0
    while True:
        pending = []
        for state in states:
            if len(state["variants"]) >= args.naug + 1 or state["attempts"] >= max_attempts:
                continue
            mode = state["modes"][min(state["target_idx"], len(state["modes"]) - 1)]
            remaining_variants = args.naug + 1 - len(state["variants"])
            remaining_attempts = max_attempts - state["attempts"]
            request_n = min(
                remaining_variants,
                remaining_attempts,
                sample_batch_size,
                max(1, consecutive_mode_quota(state["modes"], state["target_idx"])),
            )
            text = rewrite_prompt(tokenizer, state["prompt"], state["candidate"], mode)
            pending.append((request_n, state, text))
        if not pending:
            break

        for request_n in sorted({item[0] for item in pending}):
            same_n = [item for item in pending if item[0] == request_n]
            sampling = GenerationConfig(
                n=request_n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + round_idx,
            )
            for chunk in batched(same_n, rewrite_prompt_batch_size):
                outputs = generator.generate([item[2] for item in chunk], sampling, batch_size=args.batch_size)
                for (_request_n, state, _text), out_list in zip(chunk, outputs):
                    state["attempts"] += len(out_list)
                    for out in out_list:
                        out = out.strip()
                        if (
                            valid_rewrite(
                                state["candidate"],
                                out,
                                strict_numbers=not args.no_strict_numbers,
                            )
                            and out not in state["variants"]
                        ):
                            state["variants"].append(out)
                            state["target_idx"] += 1
                            if len(state["variants"]) >= args.naug + 1:
                                break
        round_idx += 1

    return [
        finalize_row(row, lookup, args, model_path, backend, sample_batch_size, rewrite_prompt_batch_size, row_batch_size)
        for row, lookup in contexts
    ]


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    existing_ids = set()
    if out_path.exists():
        existing_ids = {str(row["prompt_id"]) for row in read_jsonl(out_path)}
    candidate_rows = [row for row in read_jsonl(args.candidates) if str(row["prompt_id"]) not in existing_ids]
    if not candidate_rows:
        print(f"All candidates already perturbed in {out_path}")
        return

    model_path = resolve_model(args.rewriter_model, cache_dir=args.cache_dir, download=args.download)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    backend = pick_backend(args.backend)
    generator = make_generator(
        model_path=model_path,
        backend=backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    sample_batch_size = max(1, args.sample_batch_size)
    rewrite_prompt_batch_size = max(1, args.rewrite_prompt_batch_size)
    row_batch_size = max(1, args.row_batch_size)

    with tqdm(total=len(candidate_rows), desc="perturb candidates") as progress:
        for group in batched(candidate_rows, row_batch_size):
            for row in perturb_row_group(
                group,
                args,
                tokenizer,
                generator,
                model_path,
                backend,
                sample_batch_size,
                rewrite_prompt_batch_size,
                row_batch_size,
            ):
                append_jsonl_row(out_path, row)
                progress.update(1)


if __name__ == "__main__":
    main()
