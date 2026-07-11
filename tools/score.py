from collections.abc import Sequence
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


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(err), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.ndim != 1 or len(weight) != len(err):
            raise ValueError(
                "sample weight must be one-dimensional and match y_true"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("sample weight must be finite and nonnegative")
    sse = float(np.dot(weight, err * err))
    weight_sum = float(weight.sum())
    n = int(len(err))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        sse += float(state.get("sse", 0.0))
        weight_sum += float(
            state.get("weight_sum", getattr(combine_with, "n", 0))
        )
        n += int(getattr(combine_with, "n", 0))
    value = float(np.sqrt(sse / weight_sum)) if weight_sum else 0.0
    if combine_with is _NO_COMBINE:
        return value
    return ScoreValue(value, n, {"sse": sse, "weight_sum": weight_sum})


def weighted(score: Score) -> Score:
    if score is rmse:
        return weighted_rmse
    raise ValueError(
        f"no weighted implementation registered for {getattr(score, '__name__', score)!r}"
    )


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


def quantile_pnl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    q_buy: int = -1,
    q_sell: int = 0,
    thd_buy: float = 0.0,
    thd_sell: float = 0.0,
    combine_with: Any = _NO_COMBINE,
    power: int = 0,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if len(y) != p.shape[0]:
        raise ValueError(
            f"score length mismatch: y_true={len(y)}, y_pred={p.shape[0]}"
        )
    n_quantiles = p.shape[1]
    if not -n_quantiles <= q_buy < n_quantiles:
        raise ValueError(f"q_buy={q_buy} is out of bounds for {n_quantiles} columns")
    if not -n_quantiles <= q_sell < n_quantiles:
        raise ValueError(f"q_sell={q_sell} is out of bounds for {n_quantiles} columns")

    buy = p[:, q_buy] > thd_buy
    sell = p[:, q_sell] < thd_sell
    overlap = buy & sell
    buy &= ~overlap
    sell &= ~overlap
    active = buy | sell

    side = np.zeros(len(y), dtype=float)
    side[buy] = 1.0
    side[sell] = -1.0
    signal = np.where(buy, p[:, q_buy], np.where(sell, p[:, q_sell], 0.0))
    weight = (
        np.abs(signal[active]) ** power
        if power != 0
        else np.ones(int(active.sum()))
    )

    pnl = float(np.sum(side[active] * weight * y[active]))
    n = int(active.sum())
    norm = float(np.sum(weight)) if power != 0 else n
    n_buy = int(buy.sum())
    n_sell = int(sell.sum())
    n_overlap = int(overlap.sum())
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        pnl += float(state.get("pnl", 0.0))
        n += int(getattr(combine_with, "n", 0))
        norm += float(state.get("norm", 0.0))
        n_buy += int(state.get("n_buy", 0))
        n_sell += int(state.get("n_sell", 0))
        n_overlap += int(state.get("n_overlap", 0))
    score = pnl / norm if norm else 0.0
    if ctx is not None:
        ctx["n_active"] = n
        ctx["n_buy"] = n_buy
        ctx["n_sell"] = n_sell
        ctx["n_overlap"] = n_overlap
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(
        score,
        n,
        {
            "pnl": pnl,
            "norm": norm,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_overlap": n_overlap,
        },
    )


def get_quantile_pnl(
    q_buy: int,
    q_sell: int,
    thd_buy: float,
    thd_sell: float,
    power: int = 0,
) -> Score:
    def score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
    ) -> float | ScoreValue:
        return quantile_pnl(
            y_true,
            y_pred,
            ctx,
            q_buy,
            q_sell,
            thd_buy,
            thd_sell,
            combine_with,
            power,
        )

    score.__name__ = (
        f"quantile_pnl_buy{q_buy}_gt{thd_buy:g}_sell{q_sell}_lt{thd_sell:g}"
    )
    return score


def pinball(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    quantiles: Sequence[float] = (0.5,),
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    p = np.asarray(y_pred, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    q = np.asarray(quantiles, dtype=float)
    if p.shape[1] != q.size:
        raise ValueError(
            f"y_pred has {p.shape[1]} columns for {q.size} quantiles"
        )
    err = y - p
    loss = float(np.sum(np.maximum(q * err, (q - 1.0) * err)))
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        loss += float(getattr(combine_with, "state", {}).get("loss", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = loss / (n * q.size) if n else 0.0
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"loss": loss})


def get_pinball(quantiles: Sequence[float]) -> Score:
    q = tuple(float(x) for x in quantiles)

    def score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
    ) -> float | ScoreValue:
        return pinball(y_true, y_pred, ctx, q, combine_with)

    score.__name__ = f"pinball_{'_'.join(f'{x:g}' for x in q)}"
    return score
