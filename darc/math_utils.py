from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def clipped(values: Sequence[float], lo: float = -15.0, hi: float = 15.0) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), lo, hi)


def entropic_value(values: Sequence[float], beta: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    z = -beta * arr
    m = float(np.max(z))
    log_mean_exp = m + math.log(float(np.mean(np.exp(z - m))))
    return -log_mean_exp / beta


def cvar_low(values: Sequence[float], alpha: float = 0.1) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(alpha * arr.size)))
    return float(np.mean(arr[:k]))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    ddof = 1 if arr.size > 1 else 0
    return float(np.mean(arr)), float(np.std(arr, ddof=ddof))
