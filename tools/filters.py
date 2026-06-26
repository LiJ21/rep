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
