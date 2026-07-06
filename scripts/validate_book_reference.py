"""Validate orderbook_rs block-gated depth output against an independent
reference book builder.

The reference deliberately mirrors Databento MBO semantics with the simplest
possible Python implementation (file order, dict book, A/C/M/R mutations,
T/F/N no-ops) and compares:
  * top-N book state at sampled F_LAST boundaries (via last emitted row <= row_nr)
  * final book state
  * day-total traded size and notional vs raw T records

Usage:
  python scripts/validate_book_reference.py RAW.parquet DEPTH.parquet \
      [--levels 5] [--sample-every 1000]
"""

from __future__ import annotations

import argparse
import heapq

import numpy as np
import polars as pl

UNDEF_PRICE = 9223372036854775807
F_LAST = 128


def build_reference(raw: pl.DataFrame, levels: int, sample_every: int):
    action = raw["action"].to_numpy()
    side = raw["side"].to_numpy()
    price = raw["price"].to_numpy()
    size = raw["size"].to_numpy()
    order_id = raw["order_id"].to_numpy()
    flags = raw["flags"].to_numpy()

    orders: dict[int, tuple[str, int, int]] = {}
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}

    def levels_of(s: str) -> dict[int, int]:
        return bids if s == "B" else asks

    def add(oid: int, s: str, px: int, sz: int) -> None:
        if oid == 0 or px == UNDEF_PRICE or sz == 0:
            return
        orders[oid] = (s, px, sz)
        book = levels_of(s)
        book[px] = book.get(px, 0) + sz

    def remove_qty(s: str, px: int, sz: int) -> None:
        book = levels_of(s)
        if px in book:
            book[px] -= sz
            if book[px] <= 0:
                del book[px]

    def top(n: int) -> tuple[list, list]:
        b = heapq.nlargest(n, bids.items())
        a = heapq.nsmallest(n, asks.items())
        pad = (UNDEF_PRICE, 0)
        return (b + [pad] * (n - len(b)), a + [pad] * (n - len(a)))

    samples = []
    block = 0
    n = len(action)
    for i in range(n):
        act = action[i]
        if act == "A":
            oid = order_id[i]
            if size[i] != 0 and side[i] in ("B", "A"):
                if oid in orders:
                    os_, opx, osz = orders.pop(oid)
                    remove_qty(os_, opx, osz)
                add(oid, side[i], price[i], size[i])
        elif act == "C":
            oid = order_id[i]
            if oid in orders:
                os_, opx, osz = orders[oid]
                csz = min(size[i], osz)
                remove_qty(os_, opx, csz)
                if csz == osz:
                    del orders[oid]
                else:
                    orders[oid] = (os_, opx, osz - csz)
        elif act == "M":
            oid = order_id[i]
            if side[i] in ("B", "A"):
                if oid in orders:
                    os_, opx, osz = orders.pop(oid)
                    remove_qty(os_, opx, osz)
                add(oid, side[i], price[i], size[i])
        elif act == "R":
            orders.clear()
            bids.clear()
            asks.clear()
        # T/F/N: no book change

        if flags[i] & F_LAST:
            block += 1
            if block % sample_every == 0:
                samples.append((i, *top(levels)))
    samples.append((n - 1, *top(levels)))
    return samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("depth")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--sample-every", type=int, default=1000)
    args = p.parse_args()

    raw = pl.read_parquet(
        args.raw, columns=["action", "side", "price", "size", "order_id", "flags"]
    )
    print(f"reference pass over {raw.height} raw rows ...")
    samples = build_reference(raw, args.levels, args.sample_every)
    print(f"collected {len(samples)} sampled states")

    book_cols = [
        f"{s}_{f}_{i}" for i in range(args.levels) for s in ("bid", "ask") for f in ("px", "sz")
    ]
    out = pl.read_parquet(args.depth, columns=["row_nr", "trade_sz", "trade_vwap"] + book_cols)
    out_row_nr = out["row_nr"].to_numpy()
    assert (np.diff(out_row_nr) > 0).all(), "output row_nr not strictly increasing"

    cols = {c: out[c].to_numpy() for c in book_cols}
    mismatches = 0
    checked = 0
    for row_nr, ref_bids, ref_asks in samples:
        j = int(np.searchsorted(out_row_nr, row_nr, side="right")) - 1
        if j < 0:
            empty = all(px == UNDEF_PRICE for px, _ in ref_bids + ref_asks)
            assert empty, f"reference has book state before first emitted row (row {row_nr})"
            continue
        checked += 1
        for i in range(args.levels):
            got = (
                (cols[f"bid_px_{i}"][j], cols[f"bid_sz_{i}"][j]),
                (cols[f"ask_px_{i}"][j], cols[f"ask_sz_{i}"][j]),
            )
            want = (ref_bids[i], ref_asks[i])
            if (int(got[0][0]), int(got[0][1])) != want[0] or (
                int(got[1][0]),
                int(got[1][1]),
            ) != want[1]:
                mismatches += 1
                if mismatches <= 5:
                    print(f"MISMATCH at raw row {row_nr} level {i}: got {got}, want {want}")
                break
    print(f"book states: {checked} checked, {mismatches} mismatches")

    trades = raw.filter(pl.col("action") == "T")
    raw_qty = int(trades["size"].sum())
    out_qty = int(out["trade_sz"].sum())
    print(f"traded size: raw sum(T.size)={raw_qty}, output sum(trade_sz)={out_qty}")

    priced = trades.filter(pl.col("price") != UNDEF_PRICE)
    # f64 arithmetic: i64 price*size sums overflow past ~9e18
    raw_notional = float(
        (priced["price"].cast(pl.Float64) * priced["size"].cast(pl.Float64)).sum()
    )
    tr = out.drop_nulls("trade_vwap")
    out_notional = float((tr["trade_vwap"] * tr["trade_sz"]).sum())
    rel = abs(out_notional - raw_notional) / max(abs(raw_notional), 1.0)
    print(f"trade notional rel err: {rel:.2e}")

    ok = mismatches == 0 and raw_qty == out_qty and rel < 1e-9
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
