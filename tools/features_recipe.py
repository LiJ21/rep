from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from numbers import Real
from typing import Any

import polars as pl


HalfLife = str | timedelta | int | float
SELF_CARRYOVER = "__self__"
CHILD_CARRYOVERS = "children"

__all__ = [
    "CHILD_CARRYOVERS",
    "SELF_CARRYOVER",
    "EwmaCarryover",
    "EwmaFeature",
    "ExprFeature",
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

    def apply(self, lf: pl.LazyFrame, carryover: Any | None) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover)
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
        self, lf: pl.LazyFrame, carryover: Any | None
    ) -> pl.LazyFrame:
        child_carryovers = _child_carryovers(carryover)
        for feature in self.sub_features:
            lf = feature.apply(lf, child_carryovers.get(feature.name))
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

    def apply(self, lf: pl.LazyFrame, carryover: Any | None) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover)
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
            pl.col(input_col)
            .ewm_mean_by(pl.col(time_col), half_life=_half_life(self.half_life))
            .alias(self.name)
        )
        return lf.filter(~pl.col(seed_col)) if own_carryover is not None else lf

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
        }

    def _get_own_carryover(self, df: pl.DataFrame) -> EwmaCarryover | None:
        cols = [self._col("time_ns"), self.name]
        if df.height == 0 or any(col not in df.columns for col in cols):
            return None
        time_ns, value = df.select(cols).tail(1).row(0)
        if time_ns is None or value is None:
            return None
        return EwmaCarryover(int(time_ns), float(value))

    def _col(self, suffix: str) -> str:
        return f"__{self.name}_{suffix}"


@dataclass(frozen=True, init=False)
class TradeMomentum(ExprFeature):
    half_life: HalfLife
    log: bool
    eps: float
    time: str
    unit: bool

    def __init__(
        self,
        name: str,
        half_life: HalfLife,
        log: bool = False,
        eps: float = 1e-12,
        time: str = "ts_event",
        unit: bool = False,
    ) -> None:
        buy = _trade_side_input(0, unit)
        sell = _trade_side_input(1, unit)
        buy_ewma = EwmaFeature(
            name=f"__ewma_{name}_buy",
            expr=buy,
            half_life=half_life,
            time=time,
        )
        sell_ewma = EwmaFeature(
            name=f"__ewma_{name}_sell",
            expr=sell,
            half_life=half_life,
            time=time,
        )
        expr = _diff(pl.col(buy_ewma.name), pl.col(sell_ewma.name), log, eps)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expr", expr)
        object.__setattr__(self, "sub_features", (buy_ewma, sell_ewma))
        object.__setattr__(self, "half_life", half_life)
        object.__setattr__(self, "log", log)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "unit", unit)
        self._validate_tree()

    def to_config(self) -> dict[str, Any]:
        return {
            **super().to_config(),
            "half_life": _json_ready_half_life(self.half_life),
            "log": self.log,
            "eps": self.eps,
            "time": self.time,
            "unit": self.unit,
        }


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


def _half_life(half_life: HalfLife) -> str | timedelta:
    if isinstance(half_life, bool):
        raise TypeError(
            "half_life must be a duration string, timedelta, or positive seconds"
        )
    if isinstance(half_life, Real):
        seconds = float(half_life)
        if not isfinite(seconds) or seconds <= 0:
            raise ValueError("numeric half_life is in seconds and must be positive")
        ns = round(seconds * 1_000_000_000)
        if ns <= 0:
            raise ValueError(
                "numeric half_life is in seconds and must be at least 0.5ns"
            )
        return f"{ns}ns"
    if isinstance(half_life, (str, timedelta)):
        return half_life
    raise TypeError("half_life must be a duration string, timedelta, or positive seconds")


def _trade_side_input(side: int, unit: bool) -> pl.Expr:
    if unit:
        return pl.when(pl.col("trade_side") == side).then(1.0).otherwise(0.0)
    return (
        pl.when(pl.col("trade_side") == side)
        .then(pl.col("trade_sz").cast(pl.Float64))
        .otherwise(0.0)
    )


def _diff(left: pl.Expr, right: pl.Expr, log: bool, eps: float) -> pl.Expr:
    return (left + eps).log() - (right + eps).log() if log else left - right
