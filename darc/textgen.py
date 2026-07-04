from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    n: int = 1
    temperature: float = 0.8
    top_p: float = 0.98
    max_new_tokens: int = 320
    seed: int = 7


class HFGenerator:
    def __init__(self, model_path: str, dtype: str = "bfloat16"):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts: list[str], config: GenerationConfig, batch_size: int = 8) -> list[list[str]]:
        torch.manual_seed(config.seed)
        grouped: list[list[str]] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(batch, padding=True, return_tensors="pt").to(self.model.device)
            prompt_len = enc["input_ids"].shape[1]
            out = self.model.generate(
                **enc,
                do_sample=config.temperature > 0,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                num_return_sequences=config.n,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
            for i in range(len(batch)):
                lo = i * config.n
                hi = lo + config.n
                grouped.append([x.strip() for x in decoded[lo:hi]])
        return grouped


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
    ):
        patch_transformers_v5_tokenizer_compat()
        from vllm import LLM

        kwargs = {}
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            **kwargs,
        )

    def generate(self, prompts: list[str], config: GenerationConfig, batch_size: int = 8) -> list[list[str]]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
        )
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        return [[o.text.strip() for o in out.outputs] for out in outputs]


def patch_transformers_v5_tokenizer_compat() -> None:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import vllm

        return "vllm"
    except Exception:
        return "hf"


def make_generator(
    model_path: str,
    backend: str = "auto",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
):
    chosen = pick_backend(backend)
    if chosen == "vllm":
        return VLLMGenerator(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
    if chosen == "hf":
        return HFGenerator(model_path)
    raise ValueError(f"Unknown generation backend: {backend}")
