from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io import append_jsonl_row, read_jsonl
from .math_utils import clipped
from .modelscope_utils import resolve_model
from .templates import reward_text, reward_text_for_turn


def batched(items: list[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class RewardScorer:
    def __init__(self, model_path: str, max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def score_texts(self, texts: list[str], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for batch in batched(texts, batch_size):
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                vals = logits[:, 0]
            elif logits.ndim == 2:
                vals = logits[:, -1]
            else:
                vals = logits
            scores.extend(vals.detach().float().cpu().tolist())
        return scores

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        texts = [reward_text(self.tokenizer, prompt, response) for prompt, response in pairs]
        return self.score_texts(texts, batch_size=batch_size)

    def score_text_items(self, texts: list[str], batch_size: int = 16) -> list[float]:
        return self.score_texts(texts, batch_size=batch_size)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--perturbed", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reward-model", default="skywork_reward")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--clip", type=float, default=15.0)
    p.add_argument("--download", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    existing_ids = set()
    if out_path.exists():
        existing_ids = {str(row["prompt_id"]) for row in read_jsonl(out_path)}
    perturbed_rows = [row for row in read_jsonl(args.perturbed) if str(row["prompt_id"]) not in existing_ids]
    if not perturbed_rows:
        print(f"All rows already scored in {out_path}")
        return

    model_path = resolve_model(args.reward_model, cache_dir=args.cache_dir, download=args.download)
    scorer = RewardScorer(model_path, max_length=args.max_length)
    for row in tqdm(perturbed_rows, desc="score reward"):
        scored_candidates = []
        for candidate_item in row["perturbed_candidates"]:
            texts: list[str] = []
            lengths: list[int] = []
            for turn_idx, turn_item in enumerate(candidate_item["turn_variants"]):
                variants = turn_item["variants"]
                lengths.append(len(variants))
                texts.extend(
                    reward_text_for_turn(
                        scorer.tokenizer,
                        row["turns"],
                        candidate_item["answers"],
                        turn_idx,
                        variant,
                    )
                    for variant in variants
                )
            flat_scores = clipped(scorer.score_text_items(texts, batch_size=args.batch_size), -args.clip, args.clip).tolist()
            pos = 0
            scored_turns = []
            aggregate_scores: list[float] = []
            for turn_item, n in zip(candidate_item["turn_variants"], lengths):
                scores = flat_scores[pos : pos + n]
                pos += n
                aggregate_scores.extend(scores)
                scored_turns.append(
                    {
                        "response": turn_item["response"],
                        "variants": turn_item["variants"],
                        "scores": scores,
                        "accepted_aug": turn_item.get("accepted_aug", n - 1),
                    }
                )
            scored_candidates.append(
                {
                    "candidate_id": candidate_item["candidate_id"],
                    "answers": candidate_item["answers"],
                    "scored_turns": scored_turns,
                    "scores": aggregate_scores,
                }
            )
        row["scored_candidates"] = scored_candidates
        row["reward_model"] = str(model_path)
        row["reward_format"] = "chat_history_to_current_turn"
        append_jsonl_row(out_path, row)


if __name__ == "__main__":
    main()
