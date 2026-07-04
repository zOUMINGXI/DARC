from __future__ import annotations

import os
from pathlib import Path


MODEL_IDS = {
    "llama31_8b": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "llama3_8b": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "skywork_reward": "AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2",
}


def snapshot(
    model_id: str,
    cache_dir: str | Path | None = None,
    ignore_file_pattern: list[str] | str | None = None,
) -> str:
    from modelscope import snapshot_download

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if ignore_file_pattern is not None:
        kwargs["ignore_file_pattern"] = ignore_file_pattern
    return snapshot_download(model_id, **kwargs)


def resolve_model(name_or_path: str, cache_dir: str | Path | None = None, download: bool = False) -> str:
    if Path(name_or_path).exists():
        return str(Path(name_or_path).resolve())
    model_id = MODEL_IDS.get(name_or_path, name_or_path)
    if download:
        return snapshot(model_id, cache_dir=cache_dir, ignore_file_pattern=["original/*", "*.pth"])
    return model_id


def find_existing_model(patterns: list[str], roots: list[str | Path]) -> str | None:
    for root in roots:
        root_path = Path(os.path.expanduser(str(root)))
        if not root_path.exists():
            continue
        for pattern in patterns:
            matches = sorted(root_path.rglob(pattern))
            for match in matches:
                if match.is_dir() and (match / "config.json").exists():
                    return str(match)
    return None
