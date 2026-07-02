from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from tools.data import _mask
from tools.features_recipe import (
    BuySellMomentum,
    EwmaCarryover,
    EwmaFeature,
    ExprFeature,
    PullMomentum,
    PushMomentum,
    TradeMomentum,
)
from tools.orderbook import depth_batches, depth_table_from_arrow


UNDEF_PRICE = 9_223_372_036_854_775_807
Frame = pl.DataFrame | pl.LazyFrame
Source = Frame | str | Path
__all__ = [
    "LOBFeatures",
    "BuySellMomentum",
    "EwmaCarryover",
    "EwmaFeature",
    "ExprFeature",
    "PullMomentum",
    "PushMomentum",
    "StatefulFeature",
    "TradeMomentum",
    "add_features",
    "compute_features",
    "depth_meta",
    "mbo_to_features",
    "UNDEF_PRICE",
]


def compute_features(
    lf: pl.LazyFrame,
    feature_exprs: Mapping[str, pl.Expr],
    filters: Sequence[pl.Expr] = (),
    time: str = "ts_event",
) -> pl.LazyFrame:
    return lf.filter(_mask(filters)).select(
        pl.col(time),
        *[expr.alias(name) for name, expr in feature_exprs.items()],
    )


def add_features(lf: pl.LazyFrame, feature_exprs: Mapping[str, pl.Expr]) -> pl.LazyFrame:
    return lf.with_columns([expr.alias(name) for name, expr in feature_exprs.items()])


class StatefulFeature(ABC):
    name: str

    @abstractmethod
    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def get_carryover(self, df: pl.DataFrame) -> Any | None:
        raise NotImplementedError

    def internal_cols(self) -> list[str]:
        return []

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "stateful_feature",
            "type": type(self).__name__,
        }


def depth_meta(n: int = 1) -> list[str]:
    cols = [
        "ts_event",
        "row_nr",
        "sequence",
        "publisher_id",
        "instrument_id",
        "trade_px",
        "trade_sz",
        "trade_side",
    ]
    for i in range(n):
        cols += [f"bid_px_{i}", f"bid_sz_{i}", f"bid_ct_{i}", f"ask_px_{i}", f"ask_sz_{i}", f"ask_ct_{i}"]
    return cols


class LOBFeatures:
    @staticmethod
    def book_imbalance(depth: int, log: bool = False, eps: float = 1e-12) -> pl.Expr:
        bid0 = pl.col("bid_px_0").cast(pl.Float64)
        ask0 = pl.col("ask_px_0").cast(pl.Float64)
        mid = (bid0 + ask0) / 2
        half_spread = (ask0 - bid0) / 2
        bid = LOBFeatures._inv_distance_volume("bid", depth, mid, half_spread)
        ask = LOBFeatures._inv_distance_volume("ask", depth, mid, half_spread)
        return LOBFeatures._diff(bid, ask, log, eps)

    @staticmethod
    def size_weighted_price_gap(
        depth: int,
        total_size: float,
        log: bool = False,
        eps: float = 1e-12,
    ) -> pl.Expr:
        bid0 = pl.col("bid_px_0").cast(pl.Float64)
        ask0 = pl.col("ask_px_0").cast(pl.Float64)
        bid = LOBFeatures._avg_price_for_size("bid", depth, total_size)
        ask = LOBFeatures._avg_price_for_size("ask", depth, total_size)
        return 2. * LOBFeatures._diff(ask, bid, log, eps) / (ask0 + bid0) * 1e4

    @staticmethod
    def size_weighted_avg_price(depth: int, total_size: float) -> pl.Expr:
        bid0 = pl.col("bid_px_0").cast(pl.Float64)
        ask0 = pl.col("ask_px_0").cast(pl.Float64)
        bid = LOBFeatures._avg_price_for_size("bid", depth, total_size)
        ask = LOBFeatures._avg_price_for_size("ask", depth, total_size)
        return ((bid + ask) / (bid0 + ask0)).log() * 1e4

    @staticmethod
    def _diff(left: pl.Expr, right: pl.Expr, log: bool, eps: float) -> pl.Expr:
        return (left + eps).log() - (right + eps).log() if log else left - right

    @staticmethod
    def _px(side: str, i: int) -> pl.Expr:
        return pl.col(f"{side}_px_{i}").cast(pl.Float64)

    @staticmethod
    def _sz(side: str, i: int) -> pl.Expr:
        return pl.col(f"{side}_sz_{i}").cast(pl.Float64)

    @staticmethod
    def _valid(px: pl.Expr, sz: pl.Expr) -> pl.Expr:
        return (px != UNDEF_PRICE) & (sz > 0)

    @staticmethod
    def _inv_distance_volume(side: str, depth: int, mid: pl.Expr, half_spread: pl.Expr) -> pl.Expr:
        out = pl.lit(0.0)
        for i in range(depth):
            px, sz = LOBFeatures._px(side, i), LOBFeatures._sz(side, i)
            dist = (mid - px) / half_spread if side == "bid" else (px - mid) / half_spread
            out += (
                pl.when(LOBFeatures._valid(px, sz) & (half_spread > 0) & (dist > 0))
                .then(sz / dist)
                .otherwise(0.0)
            )
        return out

    @staticmethod
    def _avg_price_for_size(side: str, depth: int, total_size: float) -> pl.Expr:
        remaining = pl.lit(float(total_size))
        qty = pl.lit(0.0)
        cost = pl.lit(0.0)
        for i in range(depth):
            px, sz = LOBFeatures._px(side, i), LOBFeatures._sz(side, i)
            take = (
                pl.when(LOBFeatures._valid(px, sz) & (remaining > 0))
                .then(pl.min_horizontal(sz, remaining))
                .otherwise(0.0)
            )
            qty += take
            cost += take * px
            remaining -= take
        return pl.when(qty > 0).then(cost / qty).otherwise(None)


LOBFeatures.BuySellMomentum = BuySellMomentum
LOBFeatures.PullMomentum = PullMomentum
LOBFeatures.PushMomentum = PushMomentum
LOBFeatures.TradeMomentum = TradeMomentum


def mbo_to_features(
    df: Source,
    feature_exprs: Mapping[str, pl.Expr] | None = None,
    filters: Sequence[pl.Expr] = (),
    l2_depth: int | None = None,
    context_cols: Sequence[str] = ("ts_event", "ts_recv", "symbol"),
    meta_cols: Sequence[str] | None = None,
    batch_size: int = 65_536,
) -> Frame | Iterator[pl.DataFrame]:
    feature_names = tuple(feature_exprs or ())
    if isinstance(df, (str, Path)):
        return _path_batches(Path(df), feature_exprs, feature_names, filters, l2_depth, context_cols, meta_cols, batch_size)

    lazy = isinstance(df, pl.LazyFrame)
    lf = df if lazy else df.lazy()

    if l2_depth is not None:
        raw = lf.collect(engine="streaming")
        out = pl.from_arrow(depth_table_from_arrow(raw.to_arrow(), levels=l2_depth))
        keep = [c for c in context_cols if c in raw.columns and c not in out.columns]
        if keep:
            # Rust emits row_nr in original input order; use it to recover raw context.
            ctx = raw.with_row_index("row_nr").select("row_nr", *keep)
            ctx = ctx.with_columns(pl.col("row_nr").cast(pl.UInt64))
            out = out.join(ctx, on="row_nr", how="left")
        lf = out.lazy()

    lf = _apply(lf, feature_exprs, feature_names, filters, meta_cols)
    return lf if lazy else lf.collect(engine="streaming")


def _path_batches(
    path: Path,
    feature_exprs: Mapping[str, pl.Expr] | None,
    feature_names: Sequence[str],
    filters: Sequence[pl.Expr],
    l2_depth: int | None,
    context_cols: Sequence[str],
    meta_cols: Sequence[str] | None,
    batch_size: int,
) -> Iterator[pl.DataFrame]:
    if l2_depth is None:
        lf = _apply(pl.scan_parquet(path), feature_exprs, feature_names, filters, meta_cols)
        yield from lf.collect_batches(chunk_size=batch_size, maintain_order=True)
        return

    context = _path_context(path, context_cols)
    for batch in depth_batches(path, levels=l2_depth):
        out = _with_context(pl.from_arrow(batch), context)
        lf = _apply(out.lazy(), feature_exprs, feature_names, filters, meta_cols)
        yield lf.collect(engine="streaming")


def _path_context(path: Path, context_cols: Sequence[str]) -> pl.DataFrame | None:
    if not context_cols:
        return None
    cols = pl.scan_parquet(path).collect_schema().names()
    keep = [c for c in context_cols if c in cols]
    if not keep:
        return None
    return (
        pl.scan_parquet(path)
        .select(keep)
        .collect(engine="streaming")
        .with_row_index("row_nr")
        .with_columns(pl.col("row_nr").cast(pl.UInt64))
    )


def _with_context(df: pl.DataFrame, context: pl.DataFrame | None) -> pl.DataFrame:
    if context is None:
        return df
    keep = [c for c in context.columns if c != "row_nr" and c not in df.columns]
    if not keep:
        return df
    ctx = context.gather(df.get_column("row_nr").to_list()).select(keep)
    return df.hstack(ctx)


def _apply(
    lf: pl.LazyFrame,
    feature_exprs: Mapping[str, pl.Expr] | None,
    feature_names: Sequence[str],
    filters: Sequence[pl.Expr],
    meta_cols: Sequence[str] | None,
) -> pl.LazyFrame:
    if filters:
        lf = lf.filter(_mask(filters))
    if feature_exprs:
        lf = add_features(lf, feature_exprs)
    if meta_cols is not None:
        lf = lf.select(*dict.fromkeys([*meta_cols, *feature_names]))
    return lf
