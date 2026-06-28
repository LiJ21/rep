from datetime import time

import polars as pl

from tools.data import BUY, SELL


def trade_size(
    thd: float,
    trd_sz: str = "trade_sz",
    trd_side: str = "trade_side",
    bid: str = "bid_sz_0",
    ask: str = "ask_sz_0",
):
    def large_vs(col: str) -> pl.Expr:
        size = pl.col(trd_sz).cast(pl.Float64)
        depth = pl.col(col).cast(pl.Float64)
        return (depth > 0) & (size / depth > thd)

    return (
        pl.when(pl.col(trd_side) == BUY)
        .then(large_vs(ask))
        .when(pl.col(trd_side) == SELL)
        .then(large_vs(bid))
        .otherwise(False)
    )


def level_taken(
    bid: str = "bid_px_0",
    ask: str = "ask_px_0",
    eps: float = 1e-10,
):
    bid_move = pl.col(bid).diff().abs().fill_null(0) > eps
    ask_move = pl.col(ask).diff().abs().fill_null(0) > eps
    return bid_move | ask_move


def tight_spread(
    ticksize: int,
    bid: str = "bid_px_0",
    ask: str = "ask_px_0",
) -> pl.Expr:
    if ticksize < 0:
        raise ValueError("ticksize must be non-negative")
    return pl.col(ask).cast(pl.Int64) - pl.col(bid).cast(pl.Int64) == ticksize


def intraday_time(
    start: str | time,
    end: str | time,
    ts: str = "ts_event",
    timezone: str | None = None,
    closed: str = "left",
) -> pl.Expr:
    start_t, end_t = _parse_time(start), _parse_time(end)
    t = pl.col(ts)
    if timezone is not None:
        t = t.dt.convert_time_zone(timezone)
    t = t.dt.time()
    left, right = _bounds(t, start_t, end_t, closed)
    return (left & right) if start_t <= end_t else (left | right)


def _bounds(t: pl.Expr, start: time, end: time, closed: str) -> tuple[pl.Expr, pl.Expr]:
    if closed == "left":
        return t >= start, t < end
    if closed == "right":
        return t > start, t <= end
    if closed == "both":
        return t >= start, t <= end
    if closed == "none":
        return t > start, t < end
    raise ValueError("closed must be one of: 'left', 'right', 'both', 'none'")


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("time must be HH:MM or HH:MM:SS")
    hour, minute = map(int, parts[:2])
    second = int(parts[2]) if len(parts) == 3 else 0
    return time(hour, minute, second)
