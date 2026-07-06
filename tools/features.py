from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from tools.data import _mask
from tools.features_recipe import (
    CarryForwardFeature,
    EwmaCarryover,
    EwmaFeature,
    ExprFeature,
    HalfLife,
)
from tools.orderbook import depth_batches, depth_table_from_arrow
from tools.precision import check_precision, float_dtype, float_lit

UNDEF_PRICE = 9_223_372_036_854_775_807
Frame = pl.DataFrame | pl.LazyFrame
Source = Frame | str | Path
__all__ = [
    "LOBFeatures",
    "BuySellMomentum",
    "CarryForwardFeature",
    "EwmaCarryover",
    "EwmaSpread",
    "EwmaFeature",
    "ExprFeature",
    "LogReturn",
    "PullMomentum",
    "PushMomentum",
    "StatefulFeature",
    "TradeCorrelation",
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


def add_features(
    lf: pl.LazyFrame, feature_exprs: Mapping[str, pl.Expr]
) -> pl.LazyFrame:
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
        cols += [
            f"bid_px_{i}",
            f"bid_sz_{i}",
            f"bid_ct_{i}",
            f"ask_px_{i}",
            f"ask_sz_{i}",
            f"ask_ct_{i}",
        ]
    return cols


class LOBFeatures:
    @staticmethod
    def book_imbalance(
        depth: int,
        log: bool = False,
        eps: float = 1e-12,
        precision: str = "float64",
    ) -> pl.Expr:
        precision = check_precision(precision)
        bid0 = LOBFeatures._bid0(precision)
        ask0 = LOBFeatures._ask0(precision)
        mid = (bid0 + ask0) / float_lit(2.0, precision)
        half_spread = (ask0 - bid0) / float_lit(2.0, precision)
        bid = LOBFeatures._inv_distance_volume(
            "bid", depth, mid, half_spread, precision
        )
        ask = LOBFeatures._inv_distance_volume(
            "ask", depth, mid, half_spread, precision
        )
        return LOBFeatures._diff(bid, ask, log, eps, precision).cast(
            float_dtype(precision)
        )

    @staticmethod
    def size_weighted_price_gap(
        depth: int,
        total_size: float,
        log: bool = False,
        eps: float = 1e-12,
        precision: str = "float64",
    ) -> pl.Expr:
        precision = check_precision(precision)
        bid = LOBFeatures._avg_price_for_size("bid", depth, total_size, precision)
        ask = LOBFeatures._avg_price_for_size("ask", depth, total_size, precision)
        return (
            float_lit(2.0, precision)
            * LOBFeatures._diff(ask, bid, log, eps, precision)
            * float_lit(1e4, precision)
        ).cast(float_dtype(precision))

    @staticmethod
    def size_weighted_avg_price(
        depth: int,
        total_size: float,
        precision: str = "float64",
    ) -> pl.Expr:
        precision = check_precision(precision)
        bid0 = LOBFeatures._bid0(precision)
        ask0 = LOBFeatures._ask0(precision)
        bid = LOBFeatures._avg_price_for_size("bid", depth, total_size, precision)
        ask = LOBFeatures._avg_price_for_size("ask", depth, total_size, precision)
        return (((bid + ask) / (bid0 + ask0)).log() * float_lit(1e4, precision)).cast(
            float_dtype(precision)
        )

    @dataclass(frozen=True, init=False)
    class BuySellMomentum(ExprFeature):
        mode: str
        half_life: HalfLife
        log: bool
        eps: float
        time: str
        unit: bool
        normalized: bool
        combine: str
        precision: str

        def __init__(
            self,
            name: str,
            half_life: HalfLife,
            mode: str = "trade",
            log: bool = False,
            eps: float = 1e-12,
            time: str = "ts_event",
            unit: bool = False,
            normalized: bool = False,
            combine: str = "diff",
            precision: str = "float64",
        ) -> None:
            mode = LOBFeatures._check_momentum_mode(mode)
            combine = LOBFeatures._check_momentum_combine(combine)
            precision = check_precision(precision)
            buy, sell = LOBFeatures._momentum_inputs(mode, unit, precision)
            child_time = time if mode == "trade" else self._col_for(name, "time_ns")
            buy_ewma = EwmaFeature(
                name=f"__ewma_{name}_buy",
                expr=buy,
                half_life=half_life,
                time=child_time,
                normalized=normalized,
                precision=precision,
            )
            sell_ewma = EwmaFeature(
                name=f"__ewma_{name}_sell",
                expr=sell,
                half_life=half_life,
                time=child_time,
                normalized=normalized,
                precision=precision,
            )
            bbo_vol_ewma = EwmaFeature(
                name=f"__ewma_{name}_bbo_vol",
                expr=LOBFeatures._bbo_vol(precision),
                half_life=half_life,
                time=child_time,
                normalized=True,
                precision=precision,
            )
            bbo_vol = pl.max_horizontal(
                pl.col(bbo_vol_ewma.name), float_lit(eps, precision)
            )
            expr = LOBFeatures._combine_momentum_flows(
                pl.col(buy_ewma.name) / bbo_vol,
                pl.col(sell_ewma.name) / bbo_vol,
                combine,
                log,
                eps,
                precision,
            ).cast(float_dtype(precision))

            object.__setattr__(self, "name", name)
            object.__setattr__(self, "expr", expr)
            object.__setattr__(
                self, "sub_features", (buy_ewma, sell_ewma, bbo_vol_ewma)
            )
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "half_life", half_life)
            object.__setattr__(self, "log", log)
            object.__setattr__(self, "eps", eps)
            object.__setattr__(self, "time", time)
            object.__setattr__(self, "unit", unit)
            object.__setattr__(self, "normalized", normalized)
            object.__setattr__(self, "combine", combine)
            object.__setattr__(self, "precision", precision)
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
            return list(dict.fromkeys(cols))

        def to_config(self) -> dict[str, Any]:
            return {
                **super().to_config(),
                "mode": self.mode,
                "half_life": LOBFeatures._json_ready_half_life(self.half_life),
                "log": self.log,
                "eps": self.eps,
                "time": self.time,
                "unit": self.unit,
                "normalized": self.normalized,
                "combine": self.combine,
                "precision": self.precision,
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
            normalized: bool = False,
            combine: str = "diff",
            precision: str = "float64",
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
                combine=combine,
                precision=precision,
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
            normalized: bool = False,
            combine: str = "diff",
            precision: str = "float64",
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
                combine=combine,
                precision=precision,
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
            normalized: bool = False,
            combine: str = "diff",
            precision: str = "float64",
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
                combine=combine,
                precision=precision,
            )

    @dataclass(frozen=True, init=False)
    class TradeCorrelation(ExprFeature):
        mode: str
        half_life: HalfLife
        time: str
        eps: float
        normalized: bool
        precision: str

        def __init__(
            self,
            name: str,
            half_life: HalfLife,
            mode: str = "side",
            time: str = "ts_event",
            eps: float = 1e-12,
            precision: str = "float64",
        ) -> None:
            mode = LOBFeatures._check_trade_correlation_mode(mode)
            precision = check_precision(precision)
            trade_row = LOBFeatures._trade_row(precision)
            value = LOBFeatures._signed_trade_value(mode, precision)
            v_prev = CarryForwardFeature(
                name=f"__carry_{name}_v_prev",
                expr=pl.when(trade_row).then(value).otherwise(None),
                precision=precision,
            )
            pair = trade_row & pl.col(v_prev.name).is_not_null()
            v = pl.when(pair).then(value).otherwise(None)
            m_v = EwmaFeature(
                name=f"__ewma_{name}_v",
                expr=v,
                half_life=half_life,
                time=time,
                normalized=True,
                precision=precision,
            )
            m_vv = EwmaFeature(
                name=f"__ewma_{name}_vv",
                expr=pl.when(pair)
                .then(pl.col(v_prev.name) * value)
                .otherwise(None),
                half_life=half_life,
                time=time,
                normalized=True,
                precision=precision,
            )
            pair_mass = EwmaFeature(
                name=f"__ewma_{name}_pair_mass",
                expr=pl.when(pair)
                .then(float_lit(1.0, precision))
                .otherwise(float_lit(0.0, precision)),
                half_life=half_life,
                time=time,
                normalized=False,
                precision=precision,
            )
            sub_features: tuple[Any, ...] = (v_prev, m_v, m_vv)
            if mode == "volume":
                m_v2 = EwmaFeature(
                    name=f"__ewma_{name}_v2",
                    expr=pl.when(pair).then(value * value).otherwise(None),
                    half_life=half_life,
                    time=time,
                    normalized=True,
                    precision=precision,
                )
                sub_features = (*sub_features, m_v2)
                s2 = pl.col(m_v2.name)
            else:
                s2 = float_lit(1.0, precision)
            sub_features = (*sub_features, pair_mass)

            m = pl.col(m_v.name)
            q = pl.col(m_vv.name)
            variance = s2 - (m * m)
            rho = ((q - (m * m)) / (variance + float_lit(eps, precision))).clip(
                lower_bound=-1.0,
                upper_bound=1.0,
            )
            expr = (
                pl.when(pl.col(pair_mass.name) > float_lit(eps, precision))
                .then(rho)
                .otherwise(None)
                .cast(float_dtype(precision))
            )

            object.__setattr__(self, "name", name)
            object.__setattr__(self, "expr", expr)
            object.__setattr__(self, "sub_features", sub_features)
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "half_life", half_life)
            object.__setattr__(self, "time", time)
            object.__setattr__(self, "eps", eps)
            object.__setattr__(self, "normalized", True)
            object.__setattr__(self, "precision", precision)
            self._validate_tree()

        def to_config(self) -> dict[str, Any]:
            return {
                **super().to_config(),
                "mode": self.mode,
                "half_life": LOBFeatures._json_ready_half_life(self.half_life),
                "time": self.time,
                "eps": self.eps,
                "normalized": self.normalized,
                "precision": self.precision,
            }

    class LogReturn(EwmaFeature):
        def __init__(
            self,
            name: str,
            half_life: HalfLife,
            time: str = "ts_event",
            normalized: bool = False,
            precision: str = "float64",
        ) -> None:
            precision = check_precision(precision)
            valid = LOBFeatures._bbo_valid(precision)
            mid = pl.when(valid).then(LOBFeatures._mid(precision)).otherwise(None)
            expr = (mid / mid.shift(1)).log() * float_lit(1e4, precision)
            super().__init__(
                name=name,
                expr=expr,
                half_life=half_life,
                time=time,
                normalized=normalized,
                precision=precision,
            )

    class EwmaSpread(EwmaFeature):
        def __init__(
            self,
            name: str,
            half_life: HalfLife,
            time: str = "ts_event",
            precision: str = "float64",
        ) -> None:
            precision = check_precision(precision)
            valid = LOBFeatures._bbo_valid(precision)
            mid = LOBFeatures._mid(precision)
            spread = (LOBFeatures._ask0(precision) - LOBFeatures._bid0(precision)) / mid
            expr = (
                pl.when(valid & (mid > 0))
                .then(spread * float_lit(1e4, precision))
                .otherwise(None)
            )
            super().__init__(
                name=name,
                expr=expr,
                half_life=half_life,
                time=time,
                normalized=True,
                precision=precision,
            )

    @staticmethod
    def _diff(
        left: pl.Expr,
        right: pl.Expr,
        log: bool,
        eps: float,
        precision: str,
    ) -> pl.Expr:
        eps_expr = float_lit(eps, precision)
        return (
            (left + eps_expr).log() - (right + eps_expr).log()
            if log
            else (left - right) / (left + right + eps_expr)
        )

    @staticmethod
    def _combine_momentum_flows(
        buy: pl.Expr,
        sell: pl.Expr,
        combine: str,
        log: bool,
        eps: float,
        precision: str,
    ) -> pl.Expr:
        one = float_lit(1.0, precision)
        eps_expr = float_lit(eps, precision)
        if combine == "diff":
            return LOBFeatures._diff(one + buy, one + sell, log, eps, precision)
        if log:
            return (((one + buy) * (one + sell)) + eps_expr).log()
        return buy + sell

    @staticmethod
    def _bid0(precision: str = "float64") -> pl.Expr:
        return pl.col("bid_px_0").cast(float_dtype(precision))

    @staticmethod
    def _ask0(precision: str = "float64") -> pl.Expr:
        return pl.col("ask_px_0").cast(float_dtype(precision))

    @staticmethod
    def _mid(precision: str = "float64") -> pl.Expr:
        return (LOBFeatures._bid0(precision) + LOBFeatures._ask0(precision)) / float_lit(
            2.0, precision
        )

    @staticmethod
    def _bbo_vol(precision: str = "float64") -> pl.Expr:
        return float_lit(0.5, precision) * (
            pl.col("bid_sz_0")
            .cast(float_dtype(precision))
            .fill_null(float_lit(0.0, precision))
            + pl.col("ask_sz_0")
            .cast(float_dtype(precision))
            .fill_null(float_lit(0.0, precision))
        )

    @staticmethod
    def _bbo_valid(precision: str = "float64") -> pl.Expr:
        return (
            LOBFeatures._valid(
                LOBFeatures._bid0(precision),
                pl.col("bid_sz_0").cast(float_dtype(precision)),
            )
            & LOBFeatures._valid(
                LOBFeatures._ask0(precision),
                pl.col("ask_sz_0").cast(float_dtype(precision)),
            )
        ).fill_null(False)

    @staticmethod
    def _px(side: str, i: int, precision: str = "float64") -> pl.Expr:
        return pl.col(f"{side}_px_{i}").cast(float_dtype(precision))

    @staticmethod
    def _sz(side: str, i: int, precision: str = "float64") -> pl.Expr:
        return pl.col(f"{side}_sz_{i}").cast(float_dtype(precision))

    @staticmethod
    def _valid(px: pl.Expr, sz: pl.Expr) -> pl.Expr:
        return (px != UNDEF_PRICE) & (sz > 0)

    @staticmethod
    def _inv_distance_volume(
        side: str,
        depth: int,
        mid: pl.Expr,
        half_spread: pl.Expr,
        precision: str,
    ) -> pl.Expr:
        out = float_lit(0.0, precision)
        for i in range(depth):
            px, sz = LOBFeatures._px(side, i, precision), LOBFeatures._sz(
                side, i, precision
            )
            dist = (
                (mid - px) / half_spread if side == "bid" else (px - mid) / half_spread
            )
            out += (
                pl.when(LOBFeatures._valid(px, sz) & (half_spread > 0) & (dist > 0))
                .then(sz / dist)
                .otherwise(float_lit(0.0, precision))
            )
        return out

    @staticmethod
    def _avg_price_for_size(
        side: str,
        depth: int,
        total_size: float,
        precision: str,
    ) -> pl.Expr:
        remaining = float_lit(float(total_size), precision)
        qty = float_lit(0.0, precision)
        cost = float_lit(0.0, precision)
        for i in range(depth):
            px, sz = LOBFeatures._px(side, i, precision), LOBFeatures._sz(
                side, i, precision
            )
            take = (
                pl.when(LOBFeatures._valid(px, sz) & (remaining > 0))
                .then(pl.min_horizontal(sz, remaining))
                .otherwise(float_lit(0.0, precision))
            )
            qty += take
            cost += take * px
            remaining -= take
        return pl.when(qty > 0).then(cost / qty).otherwise(None)

    @staticmethod
    def _check_momentum_mode(mode: str) -> str:
        if mode not in {"trade", "push", "pull"}:
            raise ValueError("momentum mode must be one of: 'trade', 'push', 'pull'")
        return mode

    @staticmethod
    def _check_momentum_combine(combine: str) -> str:
        if combine not in {"diff", "sum"}:
            raise ValueError("momentum combine must be one of: 'diff', 'sum'")
        return combine

    @staticmethod
    def _check_trade_correlation_mode(mode: str) -> str:
        if mode not in {"side", "volume"}:
            raise ValueError("trade correlation mode must be one of: 'side', 'volume'")
        return mode

    @staticmethod
    def _momentum_inputs(
        mode: str,
        unit: bool,
        precision: str,
    ) -> tuple[pl.Expr, pl.Expr]:
        if mode == "trade":
            return (
                LOBFeatures._trade_side_input(0, unit, precision),
                LOBFeatures._trade_side_input(1, unit, precision),
            )
        if mode == "push":
            return (
                LOBFeatures._book_push_input("bid", unit, precision),
                LOBFeatures._book_push_input("ask", unit, precision),
            )
        if mode == "pull":
            return (
                LOBFeatures._book_pull_input("ask", unit, precision),
                LOBFeatures._book_pull_input("bid", unit, precision),
            )
        raise ValueError("momentum mode must be one of: 'trade', 'push', 'pull'")

    @staticmethod
    def _trade_side_input(side: int, unit: bool, precision: str) -> pl.Expr:
        if unit:
            return (
                pl.when(pl.col("trade_side") == side)
                .then(float_lit(1.0, precision))
                .otherwise(float_lit(0.0, precision))
            )
        return (
            pl.when(pl.col("trade_side") == side)
            .then(pl.col("trade_sz").cast(float_dtype(precision)))
            .otherwise(float_lit(0.0, precision))
        )

    @staticmethod
    def _trade_row(precision: str = "float64") -> pl.Expr:
        trade_size = (
            pl.col("trade_sz")
            .cast(float_dtype(precision))
            .fill_null(float_lit(0.0, precision))
        )
        return (pl.col("trade_side").is_in([0, 1]) & (trade_size > 0)).fill_null(
            False
        )

    @staticmethod
    def _trade_side_sign(precision: str) -> pl.Expr:
        side = pl.col("trade_side").cast(float_dtype(precision))
        return (
            pl.when(pl.col("trade_side").is_in([0, 1]))
            .then(float_lit(1.0, precision) - float_lit(2.0, precision) * side)
            .otherwise(None)
        )

    @staticmethod
    def _signed_trade_value(mode: str, precision: str) -> pl.Expr:
        sign = LOBFeatures._trade_side_sign(precision)
        if mode == "side":
            return sign
        if mode == "volume":
            return sign * pl.col("trade_sz").cast(float_dtype(precision)).log1p()
        raise ValueError("trade correlation mode must be one of: 'side', 'volume'")

    @staticmethod
    def _book_push_input(side: str, unit: bool, precision: str) -> pl.Expr:
        px = LOBFeatures._px(side, 0, precision)
        sz = LOBFeatures._sz(side, 0, precision).fill_null(
            float_lit(0.0, precision)
        )
        prev_px = px.shift(1)
        prev_sz = sz.shift(1).fill_null(float_lit(0.0, precision))
        diff = sz - prev_sz
        current_valid = LOBFeatures._valid(px, sz).fill_null(False)
        previous_valid = LOBFeatures._valid(prev_px, prev_sz).fill_null(False)
        same_price = current_valid & previous_valid & (px == prev_px)
        more_aggressive = px > prev_px if side == "bid" else px < prev_px
        trade_row = LOBFeatures._trade_row(precision)
        event = (
            pl.when(~trade_row & current_valid & previous_valid & more_aggressive)
            .then(sz)
            .when(~trade_row & same_price & (diff > 0))
            .then(diff)
            .otherwise(float_lit(0.0, precision))
        )
        return LOBFeatures._unitize(event, unit, precision)

    @staticmethod
    def _book_pull_input(side: str, unit: bool, precision: str) -> pl.Expr:
        px = LOBFeatures._px(side, 0, precision)
        sz = LOBFeatures._sz(side, 0, precision).fill_null(
            float_lit(0.0, precision)
        )
        prev_px = px.shift(1)
        prev_sz = sz.shift(1).fill_null(float_lit(0.0, precision))
        diff = sz - prev_sz
        current_valid = LOBFeatures._valid(px, sz).fill_null(False)
        previous_valid = LOBFeatures._valid(prev_px, prev_sz).fill_null(False)
        same_price = previous_valid & (px == prev_px)
        less_aggressive = px < prev_px if side == "bid" else px > prev_px
        level_cancelled = previous_valid & (
            ~current_valid | (current_valid & less_aggressive)
        )
        trade_row = LOBFeatures._trade_row(precision)
        event = (
            pl.when(~trade_row & level_cancelled)
            .then(prev_sz)
            .when(~trade_row & same_price & (diff < 0))
            .then(-diff)
            .otherwise(float_lit(0.0, precision))
        )
        return LOBFeatures._unitize(event, unit, precision)

    @staticmethod
    def _book_size_diff(side: str, precision: str = "float64") -> pl.Expr:
        return (
            pl.col(f"{side}_sz_0")
            .cast(float_dtype(precision))
            .diff()
            .fill_null(float_lit(0.0, precision))
        )

    @staticmethod
    def _unitize(expr: pl.Expr, unit: bool, precision: str) -> pl.Expr:
        if not unit:
            return expr.cast(float_dtype(precision))
        return (
            pl.when(expr > 0)
            .then(float_lit(1.0, precision))
            .otherwise(float_lit(0.0, precision))
        )

    @staticmethod
    def _json_ready_half_life(half_life: HalfLife) -> str | int | float:
        if isinstance(half_life, timedelta):
            return half_life.total_seconds()
        return half_life


BuySellMomentum = LOBFeatures.BuySellMomentum
EwmaSpread = LOBFeatures.EwmaSpread
LogReturn = LOBFeatures.LogReturn
PullMomentum = LOBFeatures.PullMomentum
PushMomentum = LOBFeatures.PushMomentum
TradeCorrelation = LOBFeatures.TradeCorrelation
TradeMomentum = LOBFeatures.TradeMomentum


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
        return _path_batches(
            Path(df),
            feature_exprs,
            feature_names,
            filters,
            l2_depth,
            context_cols,
            meta_cols,
            batch_size,
        )

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
        lf = _apply(
            pl.scan_parquet(path), feature_exprs, feature_names, filters, meta_cols
        )
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
