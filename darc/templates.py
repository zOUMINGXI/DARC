from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def messages_for_turns(turns: list[str], answers_so_far: list[str] | None = None) -> list[dict]:
    answers_so_far = answers_so_far or []
    messages: list[dict] = []
    for i, user_turn in enumerate(turns):
        messages.append({"role": "user", "content": user_turn})
        if i < len(answers_so_far):
            messages.append({"role": "assistant", "content": answers_so_far[i]})
    return messages


def prompt_text(tokenizer: PreTrainedTokenizerBase, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def reward_text(tokenizer: PreTrainedTokenizerBase, prompt: str, response: str) -> str:
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def reward_text_for_turn(
    tokenizer: PreTrainedTokenizerBase,
    turns: list[str],
    answers: list[str],
    turn_idx: int,
    response: str,
) -> str:
    messages: list[dict] = []
    for i, user_turn in enumerate(turns[: turn_idx + 1]):
        messages.append({"role": "user", "content": user_turn})
        if i < turn_idx:
            messages.append({"role": "assistant", "content": answers[i]})
        else:
            messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
