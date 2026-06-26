from __future__ import annotations

import re
from collections.abc import Sequence

import polars as pl

from tools.features import Frame, LOBFeatures


__all__ = ["add_future_executable_price", "add_future_price"]

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h|d|w)")
_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_NS |= {
    "m": 60 * _NS["s"],
    "h": 3_600 * _NS["s"],
    "d": 86_400 * _NS["s"],
    "w": 604_800 * _NS["s"],
}


def add_future_price(
    df: Frame,
    expr: pl.Expr,
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float = 1.0,
    time: str = "ts_event",
    name: str = "future_price",
    by: Sequence[str] = ("publisher_id", "instrument_id"),
) -> Frame:
    horizons, weights = _weighted_horizons(horizons, weights)
    lazy = isinstance(df, pl.LazyFrame)
    lf = df if lazy else df.lazy()
    join_by = [c for c in by if c in lf.collect_schema().names()]

    base = lf.with_row_index("__fp_row").with_columns(
        pl.col(time).cast(pl.Datetime("ns")).alias("__fp_t"),
        expr.cast(pl.Float64).alias("__fp_src"),
    )
    max_t = pl.col("__fp_t").max().over(join_by) if join_by else pl.col("__fp_t").max()
    base = base.with_columns(max_t.alias("__fp_max_t"))

    out = base
    terms, future_cols, total = [], [], 0.0
    for i, (horizon, weight) in enumerate(zip(horizons, weights)):
        if weight == 0:
            continue
        col = f"__fp_{i}"
        future_cols.append(col)
        total += weight
        left = base.select(
            "__fp_row",
            *join_by,
            (pl.col("__fp_t") + pl.duration(nanoseconds=_duration_ns(horizon))).alias("__fp_target"),
            "__fp_max_t",
        ).sort([*join_by, "__fp_target"])
        right = base.select(*join_by, "__fp_t", "__fp_row", pl.col("__fp_src").alias(col)).sort(
            [*join_by, "__fp_t", "__fp_row"]
        )
        joined = left.join_asof(
            right,
            left_on="__fp_target",
            right_on="__fp_t",
            by=join_by or None,
            strategy="backward",
            check_sortedness=False,
        ).select(
            "__fp_row",
            pl.when(pl.col("__fp_target") <= pl.col("__fp_max_t")).then(pl.col(col)).alias(col),
        )
        out = out.join(joined, on="__fp_row", how="left")
        terms.append(pl.col(col) * weight)

    if not terms or total == 0:
        raise ValueError("sum of non-zero weights must not be zero")
    out = out.with_columns((sum(terms, pl.lit(0.0)) / total).alias(name))
    out = out.sort("__fp_row").drop(["__fp_row", "__fp_t", "__fp_src", "__fp_max_t", *future_cols])
    return out if lazy else out.collect(engine="streaming")


def add_future_executable_price(
    df: Frame,
    depth: int,
    total_size: float,
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float = 1.0,
    time: str = "ts_event",
    name: str = "future_executable_price",
    by: Sequence[str] = ("publisher_id", "instrument_id"),
) -> Frame:
    return add_future_price(
        df,
        LOBFeatures.size_weighted_avg_price(depth, total_size),
        horizons,
        weights,
        time,
        name,
        by,
    )


def _duration_ns(duration: str) -> int:
    pos = 0
    ns = 0.0
    for match in _DURATION_RE.finditer(duration):
        if match.start() != pos:
            raise ValueError(f"invalid duration {duration!r}")
        ns += float(match[1]) * _NS[match[2]]
        pos = match.end()
    if pos != len(duration) or ns <= 0:
        raise ValueError(f"invalid duration {duration!r}")
    return round(ns)


def _weighted_horizons(
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float,
) -> tuple[list[str], list[float]]:
    horizons = [horizons] if isinstance(horizons, str) else list(horizons)
    weights = [weights] * len(horizons) if isinstance(weights, (int, float)) else list(weights)
    if len(horizons) != len(weights) or not horizons:
        raise ValueError("horizons and weights must be non-empty and equal length")
    return horizons, weights
