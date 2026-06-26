from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

Batch = tuple[np.ndarray, np.ndarray, dict[str, Any]]


@runtime_checkable
class Transform(Protocol):
    def fit(self, data: Any) -> "Transform": ...

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame: ...


class FitSource(Protocol):
    dates: list[str]
    target: str
    features: list[str]

    def frame(self, select: bool = True) -> pl.LazyFrame: ...

    def batches(self, batch_size: int | None = None) -> Iterator[Batch]: ...

    def with_transform(self, transform: Transform) -> "FitSource": ...


@dataclass
class ComposeTransform:
    steps: list[Transform]

    def fit(self, data: Any) -> "ComposeTransform":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.transform(lf)
        return lf


def compose_transform(left: Transform | None, right: Transform) -> Transform:
    if left is None:
        return right
    if isinstance(left, ComposeTransform):
        return ComposeTransform([*left.steps, right])
    return ComposeTransform([left, right])


@dataclass
class Passthrough:
    def fit(self, data: Any) -> "Passthrough":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf


@dataclass
class FunctionTransform:
    fn: Callable[[pl.LazyFrame], pl.LazyFrame]

    def fit(self, data: Any) -> "FunctionTransform":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self.fn(lf)


@dataclass
class Standardizer:
    cols: Sequence[str]
    means: dict[str, float] | None = None
    stds: dict[str, float] | None = None
    batch_size: int | None = 200_000

    def fit(self, data: Any) -> "Standardizer":
        if hasattr(data, "batches") and hasattr(data, "features"):
            return self._fit_source(data)
        return self._fit_lazy(data)

    def _fit_lazy(self, lf: pl.LazyFrame) -> "Standardizer":
        exprs = []
        for col in self.cols:
            exprs += [pl.col(col).mean().alias(f"{col}__mean"), pl.col(col).std().alias(f"{col}__std")]
        row = lf.select(exprs).collect(engine="streaming").to_dicts()[0]
        self.means = {col: float(row[f"{col}__mean"] or 0.0) for col in self.cols}
        self.stds = {col: float(row[f"{col}__std"] or 1.0) for col in self.cols}
        self.stds = {col: std if std != 0.0 else 1.0 for col, std in self.stds.items()}
        return self

    def _fit_source(self, src: Any) -> "Standardizer":
        idx = [src.features.index(col) for col in self.cols]
        n = np.zeros(len(self.cols), dtype=np.int64)
        sums = np.zeros(len(self.cols), dtype=np.float64)
        sums2 = np.zeros(len(self.cols), dtype=np.float64)

        for x, _, _ in src.batches(self.batch_size):
            vals = np.asarray(x[:, idx], dtype=np.float64)
            if vals.size == 0:
                continue
            valid = ~np.isnan(vals)
            n += valid.sum(axis=0)
            sums += np.where(valid, vals, 0.0).sum(axis=0)
            sums2 += np.where(valid, vals * vals, 0.0).sum(axis=0)

        means = np.divide(sums, n, out=np.zeros_like(sums), where=n > 0)
        numer = sums2 - (sums * sums) / np.maximum(n, 1)
        variances = np.divide(
            np.maximum(numer, 0.0),
            n - 1,
            out=np.zeros_like(sums2),
            where=n > 1,
        )
        stds = np.sqrt(variances)

        self.means = {col: float(mean) for col, mean in zip(self.cols, means)}
        self.stds = {
            col: float(std) if std != 0.0 else 1.0
            for col, std in zip(self.cols, stds)
        }
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.means is None or self.stds is None:
            raise RuntimeError("Standardizer must be fit before transform")
        return lf.with_columns(
            [((pl.col(col) - self.means[col]) / self.stds[col]).alias(col) for col in self.cols]
        )


@dataclass
class Chain:
    steps: list[Transform]

    def fit(self, data: Any) -> "Chain":
        fitted = []
        cur = data
        for step in self.steps:
            step = step.fit(cur)
            fitted.append(step)
            if hasattr(cur, "with_transform"):
                cur = cur.with_transform(step)
            else:
                cur = step.transform(cur)
        self.steps = fitted
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.transform(lf)
        return lf
