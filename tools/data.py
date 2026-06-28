from __future__ import annotations

import os
import re
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
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
POLARS_ENGINES = {"auto", "streaming", "gpu"}


def expand_dates(
    dates: str,
    exclude_holiday: str | None = "cme",
    str_result: bool = True,
    end_date: bool = True,
):
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

    if not end_date:
        days = [d for d in days if d != end]
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


def _check_polars_engine(engine: str) -> str:
    engine = engine.lower()
    if engine not in POLARS_ENGINES:
        raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
    return engine


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
    polars_engine: str = "streaming"
    _frames: list[DateFrame] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.polars_engine = _check_polars_engine(self.polars_engine)

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
        if (
            self.cache is not None
            and self.cache_key is not None
            and self.cache_key in self.cache
        ):
            return self.cache[self.cache_key]
        batch = _to_batch(
            self.frame().collect(engine=self.polars_engine),
            self.features,
            self.target,
        )
        if self.cache is not None and self.cache_key is not None:
            self.cache[self.cache_key] = batch
        return batch

    def batches(self, batch_size: int | None = None) -> Iterator[Batch]:
        for item in self.date_frames():
            lf = self._prepare(item)
            if batch_size is None:
                yield _to_batch(
                    lf.collect(engine=self.polars_engine),
                    self.features,
                    self.target,
                )
                continue
            for df in lf.collect_batches(
                chunk_size=batch_size,
                maintain_order=True,
                engine=self.polars_engine,
            ):
                yield _to_batch(df, self.features, self.target)

    def labels(self) -> tuple[np.ndarray, dict[str, Any]]:
        parts = []
        cols = [self.target, *CTX_COLS]
        for item in self.date_frames():
            parts.append(
                self._prepare(item, cols=cols).collect(engine=self.polars_engine)
            )
        df = pl.concat(parts, how="vertical") if parts else pl.DataFrame(schema=cols)
        return df.get_column(self.target).to_numpy(), _ctx_from_df(df)

    def count(self) -> int:
        return int(
            self.frame().select(pl.len()).collect(engine=self.polars_engine).item()
        )

    def with_transform(self, transform: Any) -> "DataSource":
        from tools.transform import compose_transform

        cache_key = (
            (*self.cache_key, "transform", id(transform))
            if self.cache_key is not None
            else None
        )
        out = DataSource(
            dates=list(self.dates),
            loader=self.loader,
            target=self.target,
            features=list(self.features),
            filters=self.filters,
            transform=compose_transform(self.transform, transform),
            cache=self.cache,
            cache_key=cache_key,
            polars_engine=self.polars_engine,
        )
        out._frames = self._frames
        return out

    def _prepare(
        self, item: DateFrame, cols: Sequence[str] | None = None, select: bool = True
    ) -> pl.LazyFrame:
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
    def resolve_path(
        cls,
        d: str,
        prod: str,
        path: str = RAW_PATH,
    ) -> tuple[str, str]:
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
        return os.path.join(d_dir, fname), tag

    @classmethod
    def load_date(
        cls,
        d: str,
        prod: str,
        path: str = RAW_PATH,
        filters: Sequence[pl.Expr] = (),
        cols: Sequence[str] | None = None,
    ) -> DateFrame:
        fpath, tag = cls.resolve_path(d, prod, path)
        lf = pl.scan_parquet(fpath)
        if filters:
            lf = lf.filter(_mask(filters))
        if cols is not None:
            lf = lf.select(cols)
        return DateFrame(date=d, nature=_nature_from_tag(tag), lf=lf)

    @classmethod
    def load_dates(
        cls, dates: str | Sequence[str], prod: str, **kwargs: Any
    ) -> list[DateFrame]:
        return [cls.load_date(d, prod, **kwargs) for d in _as_dates(dates)]


@dataclass
class FeatureLoader:
    prod: str | None = None
    feature_exprs: Mapping[str, pl.Expr] = field(default_factory=dict)
    return_exprs: Mapping[str, pl.Expr] = field(default_factory=dict)
    executable_returns: Mapping[str, tuple[int, float]] = field(default_factory=dict)
    horizons: Sequence[str] | str = "1s"
    weights: Sequence[float] | float = 1.0
    l2_depth: int | None = None
    path: str = RAW_PATH
    filters: tuple[pl.Expr, ...] = ()
    context_cols: tuple[str, ...] = ("ts_event", "ts_recv", "symbol")
    meta_cols: tuple[str, ...] | None = None
    batch_size: int = 65_536
    return_time: str = "ts_event"
    return_by: tuple[str, ...] = ("publisher_id", "instrument_id")

    def __call__(
        self, dates: str | Sequence[str], prod: str | None = None
    ) -> list[DateFrame]:
        return self.load_dates(dates, prod=prod)

    def load_dates(
        self, dates: str | Sequence[str], prod: str | None = None
    ) -> list[DateFrame]:
        p = prod or self.prod
        if p is None:
            raise ValueError(
                "FeatureLoader needs prod either at construction or call time"
            )
        return [self.load_date(d, p) for d in _as_dates(dates)]

    def load_date(self, d: str, prod: str | None = None) -> DateFrame:
        from tools.features import mbo_to_features
        from tools.price import add_executable_return, add_return

        p = prod or self.prod
        if p is None:
            raise ValueError(
                "FeatureLoader needs prod either at construction or call time"
            )
        fpath, tag = Raw.resolve_path(d, p, self.path)

        if self.l2_depth is None:
            raw = pl.scan_parquet(fpath)
            lf = mbo_to_features(
                raw, self.feature_exprs, self.filters, context_cols=self.context_cols
            )
        else:
            parts = mbo_to_features(
                fpath,
                self.feature_exprs,
                self.filters,
                l2_depth=self.l2_depth,
                context_cols=self.context_cols,
                batch_size=self.batch_size,
            )
            lf = pl.concat(list(parts), how="vertical_relaxed").lazy()

        for name, expr in self.return_exprs.items():
            lf = add_return(
                lf,
                expr,
                self.horizons,
                self.weights,
                self.return_time,
                name,
                self.return_by,
            )
        for name, (depth, total_size) in self.executable_returns.items():
            lf = add_executable_return(
                lf,
                depth,
                total_size,
                self.horizons,
                self.weights,
                self.return_time,
                name,
                self.return_by,
            )
        if self.meta_cols is not None:
            cols = [
                *self.meta_cols,
                *self.feature_exprs,
                *self.return_exprs,
                *self.executable_returns,
            ]
            lf = lf.select(_ordered_unique(cols))
        return DateFrame(date=d, nature=_nature_from_tag(tag), lf=lf)


@dataclass
class MultiProductLoader:
    products: Sequence[str]
    loader: Any
    on: tuple[str, ...] = ("ts_event",)
    forward_fill: bool = True
    dedup_on: bool = True
    prefix_sep: str = "__"

    def __call__(self, dates: str | Sequence[str]) -> list[DateFrame]:
        return self.load_dates(dates)

    def load_dates(self, dates: str | Sequence[str]) -> list[DateFrame]:
        ds = _as_dates(dates)
        loaded = {prod: _load_product(self.loader, ds, prod) for prod in self.products}
        by_prod = {
            prod: {item.date: item for item in frames}
            for prod, frames in loaded.items()
        }
        return [self._join_date(d, by_prod) for d in ds]

    def _join_date(
        self, d: str, by_prod: Mapping[str, Mapping[str, DateFrame]]
    ) -> DateFrame:
        items = [by_prod[prod][d] for prod in self.products]
        frames = [
            _prefix_non_keys(
                _dedup_on(item.lf, self.on) if self.dedup_on else item.lf,
                prod,
                self.on,
                self.prefix_sep,
            )
            for prod, item in zip(self.products, items)
        ]
        lf = frames[0]
        for other in frames[1:]:
            lf = lf.join(other, on=list(self.on), how="full", coalesce=True)
        lf = lf.sort(list(self.on))
        if self.forward_fill:
            lf = lf.with_columns(pl.all().exclude(self.on).forward_fill())
        nature = "+".join(_ordered_unique([item.nature for item in items]))
        return DateFrame(date=d, nature=nature, lf=lf)


def _load_product(loader: Any, dates: Sequence[str], prod: str) -> list[DateFrame]:
    fn = loader.load_dates if hasattr(loader, "load_dates") else loader
    sig = inspect.signature(fn)
    accepts_prod = "prod" in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    )
    return fn(dates, prod=prod) if accepts_prod else fn(dates)


def _prefix_non_keys(
    lf: pl.LazyFrame, prod: str, keys: Sequence[str], sep: str
) -> pl.LazyFrame:
    key_set = set(keys)
    prefix = f"{prod.lower()}{sep}"
    return lf.rename(
        {c: f"{prefix}{c}" for c in lf.collect_schema().names() if c not in key_set}
    )


def _dedup_on(lf: pl.LazyFrame, keys: Sequence[str]) -> pl.LazyFrame:
    return lf.sort(list(keys)).group_by(list(keys), maintain_order=True).last()
