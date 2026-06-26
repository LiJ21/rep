from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from tools.data import DataSource, Loader
from tools.search import (
    BySizeRecency,
    ParamFn,
    SearchSpace,
    create_study,
    expanding_folds,
    suggest_params,
    weighted_mean,
)
from tools.track import NullTracker, Tracker
from tools.transform import Passthrough, Transform


Score = Callable[[np.ndarray, np.ndarray, dict[str, Any] | None], float]


def rmse(y_true: np.ndarray, y_pred: np.ndarray, ctx: dict[str, Any] | None = None) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class Pipeline:
    rolling_dates: list[list[str]]
    test_dates: list[str]
    adapter: Any
    target: str
    features: list[str]
    data_loader: Loader
    search_space: SearchSpace | ParamFn | None = None
    val_score: Score = rmse
    test_score: Score = rmse
    transform: Transform = field(default_factory=Passthrough)
    train_filters: tuple[pl.Expr, ...] = ()
    val_filters: tuple[pl.Expr, ...] = ()
    test_filters: tuple[pl.Expr, ...] = ()
    fold_weighting: BySizeRecency = field(default_factory=BySizeRecency)
    sampler: str = "tpe"
    n_trials: int = 50
    pruner: Any = None
    tracker: Tracker = field(default_factory=NullTracker)
    cache_arrays: bool = False
    seed: int | None = None

    model: Any = field(default=None, init=False)
    best_params: dict[str, Any] | None = field(default=None, init=False)
    study: Any = field(default=None, init=False)
    fitted_transform: Transform | None = field(default=None, init=False)
    validation_history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.train_filters = tuple(self.train_filters)
        self.val_filters = tuple(self.val_filters)
        self.test_filters = tuple(self.test_filters)
        self._transform_cache: dict[tuple[str, ...], Transform] = {}
        self._array_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    def train(self) -> dict[str, Any]:
        folds = expanding_folds(self.rolling_dates)
        self.validation_history.clear()
        self.tracker.start_run({"sampler": self.sampler, "n_trials": self.n_trials, "n_folds": len(folds)})
        try:
            self.study = create_study(self.sampler, self.search_space, pruner=self.pruner, seed=self.seed)
            import optuna

            def objective(trial: Any) -> float:
                params = suggest_params(self.search_space, trial)
                trial.set_user_attr("params", params)
                self.tracker.log_params({f"param/{k}": v for k, v in params.items()})
                scores: list[float] = []
                sizes: list[int] = []
                for fold, (train_dates, val_dates) in enumerate(folds):
                    fitted = self._fit_transform(train_dates)
                    train_src = self._src(train_dates, self.train_filters, fitted, "train")
                    val_src = self._src(val_dates, self.val_filters, fitted, "val")
                    model = self.adapter.build(params)
                    model = self._fit_model(model, train_src, val_src, trial)
                    y_pred = self.adapter.predict(model, val_src)
                    y_true, ctx = val_src.labels()
                    ctx["fold"] = fold
                    loss = self._score(self.val_score, y_true, y_pred, ctx)
                    scores.append(loss)
                    sizes.append(int(ctx["n"]))
                    running = weighted_mean(scores, sizes, self.fold_weighting)
                    record = {
                        "trial": trial.number,
                        "fold": fold,
                        "val_score": loss,
                        "weighted_score": running,
                        "n": int(ctx["n"]),
                        "dates": ctx.get("dates"),
                        "natures": ctx.get("natures"),
                        "params": params,
                    }
                    self.validation_history.append(record)
                    self.tracker.log(
                        {
                            "val/fold_loss": loss,
                            "val/weighted_loss": running,
                            "val/fold": fold,
                            "val/n": int(ctx["n"]),
                        },
                        step=trial.number * len(folds) + fold,
                    )
                    trial.report(running, step=fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return weighted_mean(scores, sizes, self.fold_weighting)

            self.study.optimize(objective, n_trials=self.n_trials)
            self.best_params = dict(self.study.best_trial.user_attrs.get("params", self.study.best_params))
            self.model = self._refit(self.best_params)
            return {
                "best_params": self.best_params,
                "best_score": float(self.study.best_value),
                "n_trials": len(self.study.trials),
                "validation_history": self.validation_history,
            }
        finally:
            self.tracker.finish()

    def test(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before test()")
        transform = self.fitted_transform or self._fit_transform(self._all_train_dates())
        src = self._src(self.test_dates, self.test_filters, transform, "test")
        y_pred = self.adapter.predict(self.model, src)
        y_true, ctx = src.labels()
        ctx["fold"] = "test"
        loss = self._score(self.test_score, y_true, y_pred, ctx)
        return {"test_score": loss, "n": int(ctx["n"]), "ctx": ctx, "y_pred": y_pred}

    def get_model(self) -> Any:
        return self.model

    def _refit(self, params: dict[str, Any]) -> Any:
        train_dates = self._all_train_dates()
        self.fitted_transform = self._fit_transform(train_dates)
        train_src = self._src(train_dates, self.train_filters, self.fitted_transform, "final_train")
        model = self.adapter.build(params)
        return self._fit_model(model, train_src, None, None)

    def _fit_transform(self, dates: Sequence[str]) -> Transform:
        key = tuple(dates)
        if key not in self._transform_cache:
            src = self._src(dates, self.train_filters, None, "fit")
            transform = copy.deepcopy(self.transform)
            self._transform_cache[key] = transform.fit(src.frame(select=False))
        return self._transform_cache[key]

    def _src(
        self,
        dates: Sequence[str],
        filters: Sequence[pl.Expr],
        transform: Transform | None,
        role: str,
    ) -> DataSource:
        key = (role, tuple(dates), id(transform))
        return DataSource(
            dates=list(dates),
            loader=self.data_loader,
            target=self.target,
            features=self.features,
            filters=tuple(filters),
            transform=transform,
            cache=self._array_cache if self.cache_arrays else None,
            cache_key=key,
        )

    def _fit_model(self, model: Any, train: DataSource, val: DataSource | None, trial: Any | None) -> Any:
        sig = inspect.signature(self.adapter.fit)
        accepts_trial = "trial" in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_trial:
            return self.adapter.fit(model, train, val, self.tracker, trial=trial)
        return self.adapter.fit(model, train, val, self.tracker)

    def _score(self, score: Callable[..., float], y_true: np.ndarray, y_pred: np.ndarray, ctx: dict[str, Any]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        sig = inspect.signature(score)
        accepts_ctx = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()) or len(sig.parameters) >= 3
        return float(score(y_true, y_pred, ctx) if accepts_ctx else score(y_true, y_pred))

    def _all_train_dates(self) -> list[str]:
        return [date for chunk in self.rolling_dates for date in chunk]
