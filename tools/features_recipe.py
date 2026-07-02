from __future__ import annotations

import copy
import hashlib
import inspect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite, log
from numbers import Real
from typing import Any

import polars as pl


HalfLife = str | timedelta | int | float
SELF_CARRYOVER = "__self__"
CHILD_CARRYOVERS = "children"
_DURATION_RE = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|ms|s|m|h|d|w|i)")
_DURATION_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "µs": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
    "m": 60_000_000_000.0,
    "h": 3_600_000_000_000.0,
    "d": 86_400_000_000_000.0,
    "w": 604_800_000_000_000.0,
    "i": 1.0,
}

__all__ = [
    "CHILD_CARRYOVERS",
    "SELF_CARRYOVER",
    "BuySellMomentum",
    "EwmaCarryover",
    "EwmaFeature",
    "ExprFeature",
    "PullMomentum",
    "PushMomentum",
    "TradeMomentum",
]


@dataclass(frozen=True)
class EwmaCarryover:
    time_ns: int
    value: float


@dataclass(frozen=True)
class ExprFeature:
    name: str
    expr: pl.Expr
    sub_features: Sequence[Any] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sub_features", _copy_sub_features(self.sub_features)
        )
        self._validate_tree()

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        return lf.with_columns(self.expr.alias(self.name))

    def get_carryover(self, df: pl.DataFrame) -> dict[str, Any] | None:
        children = self._get_child_carryovers(df)
        return {CHILD_CARRYOVERS: children} if children else None

    def internal_cols(self) -> list[str]:
        cols: list[str] = []
        for feature in self.sub_features:
            cols.append(feature.name)
            cols.extend(feature.internal_cols())
        return _ordered_unique(cols)

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "stateful_feature",
            "type": type(self).__name__,
            "expr": _expr_config(self.expr),
            "sub_features": [
                _feature_config(feature) for feature in self.sub_features
            ],
        }

    def _apply_subfeatures(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        child_carryovers = _child_carryovers(carryover)
        for feature in self.sub_features:
            lf = _apply_feature(
                feature,
                lf,
                child_carryovers.get(feature.name),
                front_pad=front_pad,
            )
        return lf

    def _get_child_carryovers(self, df: pl.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for feature in self.sub_features:
            carryover = feature.get_carryover(df)
            if carryover is not None:
                out[feature.name] = carryover
        return out

    def _validate_tree(self) -> None:
        names = [self.name]
        for feature in self.sub_features:
            names.extend(_feature_tree_names(feature))
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(
                f"feature tree for {self.name!r} has duplicate names: {duplicates}"
            )


@dataclass(frozen=True)
class EwmaFeature(ExprFeature):
    half_life: HalfLife = 1.0
    time: str = "ts_event"
    normalized: bool = True

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        own_carryover = _own_carryover(carryover)

        seed_col = self._col("seed")
        time_col = self._col("time_ns")
        input_col = self._col("input")

        lf = lf.with_columns(
            pl.lit(False).alias(seed_col),
            pl.col(self.time)
            .cast(pl.Datetime("ns"))
            .dt.epoch("ns")
            .alias(time_col),
            self.expr.cast(pl.Float64).alias(input_col),
        )
        if own_carryover is not None and front_pad > 0:
            return self._apply_front_padded_ewm(
                lf,
                own_carryover,
                seed_col,
                time_col,
                input_col,
                front_pad,
            )
        if own_carryover is not None:
            seed = pl.DataFrame(
                {
                    seed_col: [True],
                    time_col: [own_carryover.time_ns],
                    input_col: [own_carryover.value],
                }
            ).lazy()
            lf = pl.concat([seed, lf], how="diagonal_relaxed")

        lf = lf.with_columns(
            self._ewm_expr(input_col, time_col).alias(self.name)
        )
        return lf.filter(~pl.col(seed_col)) if own_carryover is not None else lf

    def _apply_front_padded_ewm(
        self,
        lf: pl.LazyFrame,
        own_carryover: EwmaCarryover,
        seed_col: str,
        time_col: str,
        input_col: str,
        front_pad: int,
    ) -> pl.LazyFrame:
        order_col = self._col("front_order")
        seed_idx = front_pad - 1
        lf = lf.with_row_index(order_col)
        active = (
            lf.filter(pl.col(order_col) >= seed_idx)
            .with_columns(
                pl.when(pl.col(order_col) == seed_idx)
                .then(True)
                .otherwise(pl.col(seed_col))
                .alias(seed_col),
                pl.when(pl.col(order_col) == seed_idx)
                .then(pl.lit(own_carryover.time_ns))
                .otherwise(pl.col(time_col))
                .alias(time_col),
                pl.when(pl.col(order_col) == seed_idx)
                .then(pl.lit(own_carryover.value))
                .otherwise(pl.col(input_col))
                .alias(input_col),
            )
            .with_columns(self._ewm_expr(input_col, time_col).alias(self.name))
        )
        if front_pad <= 1:
            return active.drop(order_col)

        inactive = lf.filter(pl.col(order_col) < seed_idx).with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self.name)
        )
        return (
            pl.concat([inactive, active], how="diagonal_relaxed")
            .sort(order_col)
            .drop(order_col)
        )

    def get_carryover(self, df: pl.DataFrame) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        own = self._get_own_carryover(df)
        if own is not None:
            out[SELF_CARRYOVER] = own
        children = self._get_child_carryovers(df)
        if children:
            out[CHILD_CARRYOVERS] = children
        return out or None

    def internal_cols(self) -> list[str]:
        return _ordered_unique(
            [
                *super().internal_cols(),
                self._col("seed"),
                self._col("time_ns"),
                self._col("input"),
            ]
        )

    def to_config(self) -> dict[str, Any]:
        return {
            **super().to_config(),
            "half_life": _json_ready_half_life(self.half_life),
            "time": self.time,
            "normalized": self.normalized,
        }

    def _get_own_carryover(self, df: pl.DataFrame) -> EwmaCarryover | None:
        cols = [self._col("time_ns"), self.name]
        if df.height == 0 or any(col not in df.columns for col in cols):
            return None
        time_ns, value = df.select(cols).tail(1).row(0)
        if time_ns is None or value is None:
            return None
        return EwmaCarryover(int(time_ns), float(value))

    def _ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        if self.normalized:
            return self._normalized_ewm_expr(input_col, time_col)
        return self._unnormalized_ewm_expr(input_col, time_col)

    def _normalized_ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        return pl.col(input_col).ewm_mean_by(
            pl.col(time_col),
            half_life=_half_life(self.half_life),
        )

    def _unnormalized_ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        input_expr = pl.col(input_col)
        time_expr = pl.col(time_col)
        previous_time = time_expr.shift(1)
        first_at_time = previous_time.is_null() | (time_expr != previous_time)

        group_total = input_expr.sum().over(time_col)
        group_cumulative = input_expr.cum_sum().over(time_col)
        adjusted_input = (
            pl.when(first_at_time)
            .then(group_total / self._alpha_expr(time_col).fill_null(1.0))
            .otherwise(0.0)
        )
        group_end = adjusted_input.ewm_mean_by(
            time_expr,
            half_life=_half_life(self.half_life),
        )
        return group_end - (group_total - group_cumulative)

    def _alpha_expr(self, time_col: str) -> pl.Expr:
        dt = (pl.col(time_col) - pl.col(time_col).shift(1)).cast(pl.Float64)
        exponent = -log(2.0) * dt / float(_half_life_ns(self.half_life))
        return 1.0 - exponent.exp()

    def _col(self, suffix: str) -> str:
        return f"__{self.name}_{suffix}"


@dataclass(frozen=True, init=False)
class BuySellMomentum(ExprFeature):
    mode: str
    half_life: HalfLife
    log: bool
    eps: float
    time: str
    unit: bool
    normalized: bool

    def __init__(
        self,
        name: str,
        half_life: HalfLife,
        mode: str = "trade",
        log: bool = False,
        eps: float = 1e-12,
        time: str = "ts_event",
        unit: bool = False,
        normalized: bool = True,
    ) -> None:
        mode = _check_momentum_mode(mode)
        buy, sell = _momentum_inputs(mode, unit)
        child_time = time if mode == "trade" else self._col_for(name, "time_ns")
        buy_ewma = EwmaFeature(
            name=f"__ewma_{name}_buy",
            expr=buy,
            half_life=half_life,
            time=child_time,
            normalized=normalized,
        )
        sell_ewma = EwmaFeature(
            name=f"__ewma_{name}_sell",
            expr=sell,
            half_life=half_life,
            time=child_time,
            normalized=normalized,
        )
        expr = _diff(pl.col(buy_ewma.name), pl.col(sell_ewma.name), log, eps)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expr", expr)
        object.__setattr__(self, "sub_features", (buy_ewma, sell_ewma))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "half_life", half_life)
        object.__setattr__(self, "log", log)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "normalized", normalized)
        self._validate_tree()

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        if self.mode == "trade":
            return super().apply(lf, carryover, front_pad=front_pad)

        time_col = self._col("time_ns")
        lf = lf.with_columns(
            pl.col(self.time)
            .cast(pl.Datetime("ns"))
            .dt.epoch("ns")
            .alias(time_col),
        )
        return super().apply(lf, carryover, front_pad=front_pad)

    def internal_cols(self) -> list[str]:
        cols = super().internal_cols()
        if self.mode != "trade":
            cols.append(self._col("time_ns"))
        return _ordered_unique(cols)

    def to_config(self) -> dict[str, Any]:
        return {
            **super().to_config(),
            "mode": self.mode,
            "half_life": _json_ready_half_life(self.half_life),
            "log": self.log,
            "eps": self.eps,
            "time": self.time,
            "unit": self.unit,
            "normalized": self.normalized,
        }

    def _col(self, suffix: str) -> str:
        return self._col_for(self.name, suffix)

    @staticmethod
    def _col_for(name: str, suffix: str) -> str:
        return f"__{name}_{suffix}"


class TradeMomentum(BuySellMomentum):
    def __init__(
        self,
        name: str,
        half_life: HalfLife,
        log: bool = False,
        eps: float = 1e-12,
        time: str = "ts_event",
        unit: bool = False,
        normalized: bool = True,
    ) -> None:
        super().__init__(
            name,
            half_life,
            mode="trade",
            log=log,
            eps=eps,
            time=time,
            unit=unit,
            normalized=normalized,
        )


class PushMomentum(BuySellMomentum):
    def __init__(
        self,
        name: str,
        half_life: HalfLife,
        log: bool = False,
        eps: float = 1e-12,
        time: str = "ts_event",
        unit: bool = False,
        normalized: bool = True,
    ) -> None:
        super().__init__(
            name,
            half_life,
            mode="push",
            log=log,
            eps=eps,
            time=time,
            unit=unit,
            normalized=normalized,
        )


class PullMomentum(BuySellMomentum):
    def __init__(
        self,
        name: str,
        half_life: HalfLife,
        log: bool = False,
        eps: float = 1e-12,
        time: str = "ts_event",
        unit: bool = False,
        normalized: bool = True,
    ) -> None:
        super().__init__(
            name,
            half_life,
            mode="pull",
            log=log,
            eps=eps,
            time=time,
            unit=unit,
            normalized=normalized,
        )


def _child_carryovers(carryover: Any | None) -> Mapping[str, Any]:
    if not isinstance(carryover, Mapping):
        return {}
    children = carryover.get(CHILD_CARRYOVERS)
    if isinstance(children, Mapping):
        return children
    return {
        key: value
        for key, value in carryover.items()
        if key not in {SELF_CARRYOVER, CHILD_CARRYOVERS}
    }


def _own_carryover(carryover: Any | None) -> Any | None:
    if isinstance(carryover, Mapping):
        return carryover.get(SELF_CARRYOVER)
    return carryover


def _copy_sub_features(sub_features: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(copy.copy(feature) for feature in sub_features)


def _apply_feature(
    feature: Any,
    lf: pl.LazyFrame,
    carryover: Any | None,
    front_pad: int,
) -> pl.LazyFrame:
    sig = inspect.signature(feature.apply)
    if "front_pad" in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    ):
        return feature.apply(lf, carryover, front_pad=front_pad)
    return feature.apply(lf, carryover)


def _feature_tree_names(feature: Any) -> list[str]:
    names = [feature.name]
    for child in getattr(feature, "sub_features", ()):
        names.extend(_feature_tree_names(child))
    return names


def _duplicates(values: Sequence[str]) -> list[str]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _feature_config(feature: Any) -> dict[str, Any]:
    if hasattr(feature, "to_config"):
        return feature.to_config()
    return {"name": feature.name, "type": type(feature).__name__}


def _expr_config(expr: pl.Expr) -> dict[str, Any]:
    out: dict[str, Any] = {"repr": str(expr)}
    try:
        out["roots"] = expr.meta.root_names()
    except Exception:
        pass
    try:
        blob = expr.meta.serialize(format="json")
    except Exception:
        return out
    out.update(
        {
            "format": "json",
            "polars_version": pl.__version__,
            "expr": blob,
            "fingerprint": hashlib.sha256(blob.encode()).hexdigest(),
        }
    )
    return out


def _json_ready_half_life(half_life: HalfLife) -> str | int | float:
    if isinstance(half_life, timedelta):
        return half_life.total_seconds()
    return half_life


def _half_life_ns(half_life: HalfLife) -> int:
    if isinstance(half_life, bool):
        raise TypeError(
            "half_life must be a duration string, timedelta, or positive seconds"
        )
    if isinstance(half_life, Real):
        return _seconds_ns(float(half_life))
    if isinstance(half_life, timedelta):
        return _seconds_ns(half_life.total_seconds())
    if isinstance(half_life, str):
        return _duration_str_ns(half_life)
    raise TypeError("half_life must be a duration string, timedelta, or positive seconds")


def _half_life(half_life: HalfLife) -> str | timedelta:
    if isinstance(half_life, bool):
        raise TypeError(
            "half_life must be a duration string, timedelta, or positive seconds"
        )
    if isinstance(half_life, Real):
        return f"{_seconds_ns(float(half_life))}ns"
    if isinstance(half_life, (str, timedelta)):
        return half_life
    raise TypeError("half_life must be a duration string, timedelta, or positive seconds")


def _seconds_ns(seconds: float) -> int:
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("numeric half_life is in seconds and must be positive")
    ns = round(seconds * 1_000_000_000)
    if ns <= 0:
        raise ValueError("numeric half_life is in seconds and must be at least 0.5ns")
    return ns


def _duration_str_ns(duration: str) -> int:
    text = duration.strip()
    if not text:
        raise ValueError("duration string half_life must not be empty")
    total = 0.0
    pos = 0
    while pos < len(text):
        match = _DURATION_RE.match(text, pos)
        if match is None:
            raise ValueError(f"unsupported duration string half_life: {duration!r}")
        value, unit = match.groups()
        total += float(value) * _DURATION_NS[unit]
        pos = match.end()
    ns = round(total)
    if ns <= 0:
        raise ValueError("duration string half_life must be at least 0.5ns")
    return ns


def _check_momentum_mode(mode: str) -> str:
    if mode not in {"trade", "push", "pull"}:
        raise ValueError("momentum mode must be one of: 'trade', 'push', 'pull'")
    return mode


def _momentum_inputs(mode: str, unit: bool) -> tuple[pl.Expr, pl.Expr]:
    if mode == "trade":
        return _trade_side_input(0, unit), _trade_side_input(1, unit)
    if mode == "push":
        return _book_push_input("bid", unit), _book_push_input("ask", unit)
    if mode == "pull":
        return _book_pull_input("ask", unit), _book_pull_input("bid", unit)
    raise ValueError("momentum mode must be one of: 'trade', 'push', 'pull'")


def _trade_side_input(side: int, unit: bool) -> pl.Expr:
    if unit:
        return pl.when(pl.col("trade_side") == side).then(1.0).otherwise(0.0)
    return (
        pl.when(pl.col("trade_side") == side)
        .then(pl.col("trade_sz").cast(pl.Float64))
        .otherwise(0.0)
    )


def _book_push_input(side: str, unit: bool) -> pl.Expr:
    return _unitize(_book_size_diff(side).clip(lower_bound=0), unit)


def _book_pull_input(side: str, unit: bool) -> pl.Expr:
    return _unitize((-_book_size_diff(side)).clip(lower_bound=0), unit)


def _book_size_diff(side: str) -> pl.Expr:
    return pl.col(f"{side}_sz_0").cast(pl.Float64).diff().fill_null(0.0)


def _unitize(expr: pl.Expr, unit: bool) -> pl.Expr:
    if not unit:
        return expr
    return pl.when(expr > 0).then(1.0).otherwise(0.0)


def _diff(left: pl.Expr, right: pl.Expr, log: bool, eps: float) -> pl.Expr:
    return (left + eps).log() - (right + eps).log() if log else left - right
