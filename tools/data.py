from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as ec
import numpy as np
import polars as pl


_HOLIDAY_CALENDARS = {"cme": "CMES"}
_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = str(
    _ROOT
    / "data/databento_glbx_mdp3_mbo_full_day_parquet/{prod}M6_{d}_{tag}_{prod_s}_full_day.parquet"
)
CTX_COLS = ("date", "nature")

BUY = 0
SELL = 1


@dataclass(frozen=True)
class DateFrame:
    date: str
    nature: str
    lf: pl.LazyFrame


Loader = Callable[[list[str]], list[DateFrame]]
Batch = tuple[np.ndarray, np.ndarray, dict[str, Any]]


def expand_dates(dates: str, exclude_holiday: str | None = "cme", str_result: bool = True):
    parts = dates.split("-")
    start = datetime.strptime(parts[0], "%Y%m%d").date()
    end = datetime.strptime(parts[-1], "%Y%m%d").date()
    if end < start:
        raise ValueError(f"end date {parts[-1]} precedes start date {parts[0]}")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if exclude_holiday:
        cal = _HOLIDAY_CALENDARS.get(exclude_holiday.lower())
        if cal is None:
            raise ValueError(
                f"unsupported exclude_holiday {exclude_holiday!r}; supported: {sorted(_HOLIDAY_CALENDARS)}"
            )
        sessions = set(ec.get_calendar(cal).sessions_in_range(start, end).date)
        days = [d for d in days if d in sessions]

    return [d.isoformat() for d in days] if str_result else days


def _as_dates(dates: str | Sequence[str]) -> list[str]:
    return list(dates) if not isinstance(dates, str) else expand_dates(dates)


def _nature_from_tag(tag: str) -> str:
    parts = tag.lower().split("_")
    if "stress" in parts:
        return "stress"
    if "normal" in parts:
        return "normal"
    return tag


def _mask(filters: Sequence[pl.Expr]) -> pl.Expr:
    if not filters:
        return pl.lit(True)
    mask = filters[0]
    for expr in filters[1:]:
        mask = mask | expr
    return mask


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _ctx_from_df(df: pl.DataFrame) -> dict[str, Any]:
    ctx = {"n": df.height}
    for col in CTX_COLS:
        if col in df.columns:
            values = df.get_column(col).to_numpy()
            ctx[col] = values
            ctx[f"{col}s"] = _ordered_unique([str(v) for v in values.tolist()])
    return ctx


def _to_batch(df: pl.DataFrame, features: Sequence[str], target: str) -> Batch:
    x = df.select(features).to_numpy()
    y = df.get_column(target).to_numpy()
    return x, y, _ctx_from_df(df)


@dataclass
class DataSource:
    dates: list[str]
    loader: Loader
    target: str
    features: list[str]
    filters: tuple[pl.Expr, ...] = ()
    transform: Any = None
    cache: dict[tuple[Any, ...], Batch] | None = None
    cache_key: tuple[Any, ...] | None = None
    _frames: list[DateFrame] | None = field(default=None, init=False, repr=False)

    def date_frames(self) -> list[DateFrame]:
        if self._frames is None:
            self._frames = self.loader(list(self.dates))
        return self._frames

    def frame(self, select: bool = True) -> pl.LazyFrame:
        frames = [self._prepare(item, select=select) for item in self.date_frames()]
        if not frames:
            raise ValueError("DataSource has no dates")
        return pl.concat(frames, how="vertical")

    def materialize(self) -> Batch:
        if self.cache is not None and self.cache_key is not None and self.cache_key in self.cache:
            return self.cache[self.cache_key]
        batch = _to_batch(self.frame().collect(engine="streaming"), self.features, self.target)
        if self.cache is not None and self.cache_key is not None:
            self.cache[self.cache_key] = batch
        return batch

    def batches(self, batch_size: int | None = None) -> Iterator[Batch]:
        for item in self.date_frames():
            lf = self._prepare(item)
            if batch_size is None:
                yield _to_batch(lf.collect(engine="streaming"), self.features, self.target)
                continue
            for df in lf.collect_batches(chunk_size=batch_size, maintain_order=True):
                yield _to_batch(df, self.features, self.target)

    def labels(self) -> tuple[np.ndarray, dict[str, Any]]:
        parts = []
        cols = [self.target, *CTX_COLS]
        for item in self.date_frames():
            parts.append(self._prepare(item, cols=cols).collect(engine="streaming"))
        df = pl.concat(parts, how="vertical") if parts else pl.DataFrame(schema=cols)
        return df.get_column(self.target).to_numpy(), _ctx_from_df(df)

    def count(self) -> int:
        return int(self.frame().select(pl.len()).collect(engine="streaming").item())

    def _prepare(self, item: DateFrame, cols: Sequence[str] | None = None, select: bool = True) -> pl.LazyFrame:
        lf = item.lf.with_columns(
            pl.lit(item.date).alias("date"),
            pl.lit(item.nature).alias("nature"),
        )
        if self.filters:
            lf = lf.filter(_mask(self.filters))
        if self.transform is not None:
            lf = self.transform.transform(lf)
        if not select:
            return lf
        selected = cols or [*self.features, self.target, *CTX_COLS]
        return lf.select(_ordered_unique(selected))


class Raw:
    @classmethod
    def load_date(
        cls,
        d: str,
        prod: str,
        path: str = RAW_PATH,
        filters: Sequence[pl.Expr] = (),
        cols: Sequence[str] | None = None,
    ) -> DateFrame:
        pattern = path.format(
            prod=prod,
            prod_s=prod.lower(),
            d=d,
            dnd=d.replace("-", ""),
            dslash=d.replace("-", "/"),
            tag=r"(?P<tag>.+)",
        )
        d_dir, name_pat = os.path.split(pattern)
        d_dir = d_dir or "."
        regex = re.compile("^" + name_pat + "$")
        matched = []
        for fname in os.listdir(d_dir):
            match = regex.match(fname)
            if not match:
                continue
            tag = match.groupdict().get("tag")
            if tag is None and match.groups():
                tag = match.group(1)
            matched.append((fname, tag or "unknown"))
        if len(matched) != 1:
            raise ValueError(
                f"expected exactly one file matching {name_pat!r} in {d_dir!r}, "
                f"matched {[fname for fname, _ in matched]}"
            )

        fname, tag = matched[0]
        lf = pl.scan_parquet(os.path.join(d_dir, fname))
        if filters:
            lf = lf.filter(_mask(filters))
        if cols is not None:
            lf = lf.select(cols)
        return DateFrame(date=d, nature=_nature_from_tag(tag), lf=lf)

    @classmethod
    def load_dates(cls, dates: str | Sequence[str], prod: str, **kwargs: Any) -> list[DateFrame]:
        return [cls.load_date(d, prod, **kwargs) for d in _as_dates(dates)]
