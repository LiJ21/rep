from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Transform(Protocol):
    def fit(self, lf: pl.LazyFrame) -> "Transform": ...

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame: ...


@dataclass
class Passthrough:
    def fit(self, lf: pl.LazyFrame) -> "Passthrough":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf


@dataclass
class FunctionTransform:
    fn: Callable[[pl.LazyFrame], pl.LazyFrame]

    def fit(self, lf: pl.LazyFrame) -> "FunctionTransform":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self.fn(lf)


@dataclass
class Standardizer:
    cols: Sequence[str]
    means: dict[str, float] | None = None
    stds: dict[str, float] | None = None

    def fit(self, lf: pl.LazyFrame) -> "Standardizer":
        exprs = []
        for col in self.cols:
            exprs += [pl.col(col).mean().alias(f"{col}__mean"), pl.col(col).std().alias(f"{col}__std")]
        row = lf.select(exprs).collect(engine="streaming").to_dicts()[0]
        self.means = {col: float(row[f"{col}__mean"] or 0.0) for col in self.cols}
        self.stds = {col: float(row[f"{col}__std"] or 1.0) for col in self.cols}
        self.stds = {col: std if std != 0.0 else 1.0 for col, std in self.stds.items()}
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

    def fit(self, lf: pl.LazyFrame) -> "Chain":
        fitted = []
        cur = lf
        for step in self.steps:
            step = step.fit(cur)
            cur = step.transform(cur)
            fitted.append(step)
        self.steps = fitted
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.transform(lf)
        return lf
