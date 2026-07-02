from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

Score = Callable[..., float]
_NO_COMBINE = object()


@dataclass(frozen=True)
class ScoreValue:
    value: float
    n: int
    state: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    err = np.asarray(y_true) - np.asarray(y_pred)
    sse = float(np.dot(err, err))
    n = int(len(err))
    if combine_with is _NO_COMBINE:
        return float(np.sqrt(sse / n)) if n else 0.0
    if combine_with is not None:
        sse += float(getattr(combine_with, "state", {}).get("sse", 0.0))
        n += int(getattr(combine_with, "n", 0))
    return ScoreValue(float(np.sqrt(sse / n)) if n else 0.0, n, {"sse": sse})


def r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    err = y - np.asarray(y_pred, dtype=float)
    sse = float(np.dot(err, err))
    y_sum = float(y.sum())
    y2_sum = float(np.dot(y, y))
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        sse += float(state.get("sse", 0.0))
        y_sum += float(state.get("y_sum", 0.0))
        y2_sum += float(state.get("y2_sum", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = _r2_from_stats(sse, y_sum, y2_sum, n)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"sse": sse, "y_sum": y_sum, "y2_sum": y2_sum})


def _r2_from_stats(sse: float, y_sum: float, y2_sum: float, n: int) -> float:
    if n == 0:
        return 0.0
    sst = y2_sum - y_sum * y_sum / n
    if sst <= 0.0:
        return 1.0 if sse <= 0.0 else 0.0
    return float(1.0 - sse / sst)


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    state = {
        "y_sum": float(y.sum()),
        "p_sum": float(p.sum()),
        "y2_sum": float(np.dot(y, y)),
        "p2_sum": float(np.dot(p, p)),
        "yp_sum": float(np.dot(y, p)),
    }
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        old = getattr(combine_with, "state", {})
        state = {k: v + float(old.get(k, 0.0)) for k, v in state.items()}
        n += int(getattr(combine_with, "n", 0))
    score = _corr_from_stats(n, **state)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, state)


def _corr_from_stats(
    n: int,
    y_sum: float,
    p_sum: float,
    y2_sum: float,
    p2_sum: float,
    yp_sum: float,
) -> float:
    if n < 2:
        return 0.0
    cov = yp_sum - y_sum * p_sum / n
    y_var = y2_sum - y_sum * y_sum / n
    p_var = p2_sum - p_sum * p_sum / n
    denom = y_var * p_var
    return float(cov / np.sqrt(denom)) if denom > 0.0 else 0.0


def unit_pnl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    threshold: float = 0.0,
    combine_with: Any = _NO_COMBINE,
    power: int = 0,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.abs(p) > threshold
    pnl = float(np.sum(np.sign(p[mask]) * np.abs(p[mask]) ** power * y[mask]))
    n = int(len(p[mask]))
    norm = float(np.sum(np.abs(p[mask]) ** power)) if power != 0 else n
    if combine_with is not _NO_COMBINE and combine_with is not None:
        pnl += float(getattr(combine_with, "state", {}).get("pnl", 0.0))
        n += int(getattr(combine_with, "n", 0))
        norm += float(getattr(combine_with, "state", {}).get("norm", 0.0))
    score = pnl / norm if norm else 0.0
    if ctx is not None:
        ctx["n_active"] = n
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"pnl": pnl, "norm": norm})


def get_unit_pnl(threshold: float, power: int = 0) -> Score:
    def score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
    ) -> float | ScoreValue:
        return unit_pnl(y_true, y_pred, ctx, threshold, combine_with, power)

    score.__name__ = f"unit_pnl_{threshold:g}"
    return score
