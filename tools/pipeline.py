from __future__ import annotations

import copy
import gc
import inspect
import resource
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from tools.data import POLARS_ENGINES, DataSource, Loader
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

Score = Callable[..., float]
_NO_COMBINE = object()


@dataclass(frozen=True)
class ScoreValue:
    value: float
    n: int
    state: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    err = np.asarray(y_true) - np.asarray(y_pred)
    sse = float(np.dot(err, err))
    n = int(len(err))
    if combine_with is _NO_COMBINE:
        return float(np.sqrt(sse / n)) if n else 0.0
    if combine_with is not None:
        sse += float(getattr(combine_with, "state", {}).get("sse", 0.0))
        n += int(getattr(combine_with, "n", 0))
    return ScoreValue(float(np.sqrt(sse / n)) if n else 0.0, n, {"sse": sse})


def r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    err = y - np.asarray(y_pred, dtype=float)
    sse = float(np.dot(err, err))
    y_sum = float(y.sum())
    y2_sum = float(np.dot(y, y))
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        sse += float(state.get("sse", 0.0))
        y_sum += float(state.get("y_sum", 0.0))
        y2_sum += float(state.get("y2_sum", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = _r2_from_stats(sse, y_sum, y2_sum, n)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"sse": sse, "y_sum": y_sum, "y2_sum": y2_sum})


def _r2_from_stats(sse: float, y_sum: float, y2_sum: float, n: int) -> float:
    if n == 0:
        return 0.0
    sst = y2_sum - y_sum * y_sum / n
    if sst <= 0.0:
        return 1.0 if sse <= 0.0 else 0.0
    return float(1.0 - sse / sst)


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    state = {
        "y_sum": float(y.sum()),
        "p_sum": float(p.sum()),
        "y2_sum": float(np.dot(y, y)),
        "p2_sum": float(np.dot(p, p)),
        "yp_sum": float(np.dot(y, p)),
    }
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        old = getattr(combine_with, "state", {})
        state = {k: v + float(old.get(k, 0.0)) for k, v in state.items()}
        n += int(getattr(combine_with, "n", 0))
    score = _corr_from_stats(n, **state)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, state)


def _corr_from_stats(
    n: int,
    y_sum: float,
    p_sum: float,
    y2_sum: float,
    p2_sum: float,
    yp_sum: float,
) -> float:
    if n < 2:
        return 0.0
    cov = yp_sum - y_sum * p_sum / n
    y_var = y2_sum - y_sum * y_sum / n
    p_var = p2_sum - p_sum * p_sum / n
    denom = y_var * p_var
    return float(cov / np.sqrt(denom)) if denom > 0.0 else 0.0


def unit_pnl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    threshold: float = 0.0,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.abs(p) > threshold
    pnl = float(np.sum(np.sign(p[mask]) * y[mask]))
    n = int(len(p))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        pnl += float(getattr(combine_with, "state", {}).get("pnl", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = pnl / n if n else 0.0
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"pnl": pnl})


def get_unit_pnl(threshold: float) -> Score:
    def score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
    ) -> float | ScoreValue:
        return unit_pnl(y_true, y_pred, ctx, threshold, combine_with)

    score.__name__ = f"unit_pnl_{threshold:g}"
    return score


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
    score_direction: str = "minimize"
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
    polars_engine: str = "streaming"

    model: Any = field(default=None, init=False)
    best_params: dict[str, Any] | None = field(default=None, init=False)
    study: Any = field(default=None, init=False)
    fitted_transform: Transform | None = field(default=None, init=False)
    validation_history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.score_direction = self.score_direction.lower()
        if self.score_direction not in {"minimize", "maximize"}:
            raise ValueError("score_direction must be 'minimize' or 'maximize'")
        self.polars_engine = self.polars_engine.lower()
        if self.polars_engine not in POLARS_ENGINES:
            raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
        self.train_filters = tuple(self.train_filters)
        self.val_filters = tuple(self.val_filters)
        self.test_filters = tuple(self.test_filters)
        self._transform_cache: dict[tuple[Any, ...], Transform] = {}
        self._array_cache: dict[
            tuple[Any, ...], tuple[np.ndarray, np.ndarray, dict[str, Any]]
        ] = {}

    def train(
        self,
        verbose: int = 0,
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> dict[str, Any]:
        folds = expanding_folds(self.rolling_dates)
        self.validation_history.clear()
        self.tracker.start_run(
            {
                "sampler": self.sampler,
                "n_trials": self.n_trials,
                "n_folds": len(folds),
                "score_direction": self.score_direction,
                "polars_engine": self.polars_engine,
            }
        )
        try:
            self.study = create_study(
                self.sampler,
                self.search_space,
                direction=self.score_direction,
                pruner=self.pruner,
                seed=self.seed,
            )
            import optuna

            if verbose > 0:
                print(f"======== Optuna study created. Launching optimization.")

            def objective(trial: Any) -> float:
                params = suggest_params(self.search_space, trial)
                trial.set_user_attr("params", params)
                if verbose > 0:
                    print(f"======== running params {params}")
                self.tracker.log_params({f"param/{k}": v for k, v in params.items()})
                scores: list[float] = []
                sizes: list[int] = []
                for fold, (train_dates, val_dates) in enumerate(folds):
                    if verbose > 1:
                        print(
                            f"======== fold: {fold}, with train = {train_dates} and val = {val_dates}"
                        )
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _fit_transform train_dates={train_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        fitted = self._fit_transform(train_dates)
                    train_src = self._src(
                        train_dates, self.train_filters, fitted, "train"
                    )
                    val_src = self._src(val_dates, self.val_filters, fitted, "val")
                    model = self.adapter.build(params)
                    if verbose > 2:
                        print("======== built model, start training...")
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _fit_model train_dates={train_dates} val_dates={val_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        model = self._fit_model(model, train_src, val_src, trial)
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _evaluate val_dates={val_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        loss, ctx, _ = self._evaluate(
                            model, val_src, self.val_score, fold
                        )
                    scores.append(loss)
                    sizes.append(int(ctx["n"]))
                    running = weighted_mean(scores, sizes, self.fold_weighting)
                    if verbose > 1:
                        print(f"======== loss = {loss}, running average = {running}")
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
                    if verbose > 2:
                        print("======== record = \n", record)
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
            if verbose > 0:
                print(
                    "======== optimization finished, best params extracted. Refitting with best params."
                )
            self.best_params = dict(
                self.study.best_trial.user_attrs.get("params", self.study.best_params)
            )
            self.model = self._refit(
                self.best_params,
                memory_log=memory_log,
                memory_interval=memory_interval,
            )
            print("======== training done.")
            return {
                "best_params": self.best_params,
                "best_score": float(self.study.best_value),
                "n_trials": len(self.study.trials),
                "validation_history": self.validation_history,
            }
        finally:
            self.tracker.finish()

    def test(self, score: Score = rmse) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before test()")
        transform = self.fitted_transform or self._fit_transform(
            self._all_train_dates()
        )
        src = self._src(self.test_dates, self.test_filters, transform, "test")
        loss, ctx, y_pred = self._evaluate(
            self.model, src, score, "test", keep_predictions=True
        )
        return {"test_score": loss, "n": int(ctx["n"]), "ctx": ctx, "y_pred": y_pred}

    def get_model(self) -> Any:
        return self.model

    def _refit(
        self,
        params: dict[str, Any],
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> Any:
        train_dates = self._all_train_dates()
        with self._memory_log(
            f"_refit _fit_transform train_dates={train_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            self.fitted_transform = self._fit_transform(train_dates)
        train_src = self._src(
            train_dates, self.train_filters, self.fitted_transform, "final_train"
        )
        model = self.adapter.build(params)
        with self._memory_log(
            f"_refit _fit_model train_dates={train_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            return self._fit_model(model, train_src, None, None)

    def _fit_transform(self, dates: Sequence[str]) -> Transform:
        key = (tuple(dates), self.polars_engine)
        if key not in self._transform_cache:
            src = self._src(dates, self.train_filters, None, "fit")
            transform = copy.deepcopy(self.transform)
            self._transform_cache[key] = transform.fit(src)
        return self._transform_cache[key]

    def _src(
        self,
        dates: Sequence[str],
        filters: Sequence[pl.Expr],
        transform: Transform | None,
        role: str,
    ) -> DataSource:
        key = (role, tuple(dates), id(transform), self.polars_engine)
        return DataSource(
            dates=list(dates),
            loader=self.data_loader,
            target=self.target,
            features=self.features,
            filters=tuple(filters),
            transform=transform,
            cache=self._array_cache if self.cache_arrays else None,
            cache_key=key,
            polars_engine=self.polars_engine,
        )

    def _fit_model(
        self, model: Any, train: DataSource, val: DataSource | None, trial: Any | None
    ) -> Any:
        sig = inspect.signature(self.adapter.fit)
        accepts_trial = "trial" in sig.parameters or any(
            p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_trial:
            return self.adapter.fit(model, train, val, self.tracker, trial=trial)
        return self.adapter.fit(model, train, val, self.tracker)

    def _evaluate(
        self,
        model: Any,
        src: DataSource,
        score: Callable[..., float],
        fold: Any,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        if not getattr(self.adapter, "streaming", False):
            x, y_true, ctx = src.materialize()
            y_pred = self.adapter.predict(model, x)
            ctx["fold"] = fold
            return (
                self._score(score, y_true, y_pred, ctx),
                ctx,
                y_pred if keep_predictions else None,
            )
        loss, ctx, y_pred = self._score_stream(model, score, src, keep_predictions)
        ctx["fold"] = fold
        return loss, ctx, y_pred

    def _score(
        self,
        score: Callable[..., float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any],
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )
        return float(self._call_score(score, y_true, y_pred, ctx))

    def _score_stream(
        self,
        model: Any,
        score: Callable[..., float],
        src: DataSource,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        state: Any = None
        ctx: dict[str, Any] = {"n": 0}
        pred_parts = [] if keep_predictions else None

        for x, y_true, batch_ctx in src.batches(self._adapter_batch_size()):
            y_pred = np.asarray(self.adapter.predict(model, x))
            if len(y_true) != len(y_pred):
                raise ValueError(
                    f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
                )
            state = self._call_score(
                score, y_true, y_pred, batch_ctx, combine_with=state
            )
            self._merge_ctx(ctx, batch_ctx)
            if pred_parts is not None:
                pred_parts.append(y_pred)

        if state is None:
            raise ValueError("cannot score empty prediction stream")
        y_pred = np.concatenate(pred_parts) if pred_parts else None
        return float(state), ctx, y_pred

    def _call_score(
        self,
        score: Callable[..., float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any],
        combine_with: Any = _NO_COMBINE,
    ) -> Any:
        sig = inspect.signature(score)
        params = list(sig.parameters.values())
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
        positional = [
            p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        args: list[Any] = [y_true, y_pred]
        kwargs: dict[str, Any] = {}
        if has_varargs or (
            len(positional) >= 3 and positional[2].name != "combine_with"
        ):
            args.append(ctx)
        elif "ctx" in sig.parameters:
            kwargs["ctx"] = ctx
        if combine_with is not _NO_COMBINE:
            if "combine_with" not in sig.parameters and not has_varkw:
                raise TypeError("streaming scores must accept a combine_with argument")
            kwargs["combine_with"] = combine_with
        return score(*args, **kwargs)

    def _adapter_batch_size(self) -> int | None:
        return getattr(self.adapter, "batch_size", None)

    def _merge_ctx(self, total: dict[str, Any], batch: dict[str, Any]) -> None:
        total["n"] = int(total.get("n", 0)) + int(batch.get("n", 0))
        for name in ("date", "nature"):
            values = batch.get(f"{name}s")
            if values is None and name in batch:
                values = [str(v) for v in np.asarray(batch[name]).tolist()]
            if values is not None:
                key = f"{name}s"
                total[key] = _ordered_unique(
                    [*total.get(key, []), *[str(v) for v in values]]
                )

    def _all_train_dates(self) -> list[str]:
        return [date for chunk in self.rolling_dates for date in chunk]

    @contextmanager
    def _memory_log(
        self,
        label: str,
        enabled: bool = False,
        interval: float = 0.05,
    ):
        if not enabled:
            yield
            return

        try:
            import psutil
        except ImportError as exc:
            raise ImportError("memory_log=True requires psutil.") from exc

        proc = psutil.Process()
        gc.collect()
        start = proc.memory_info().rss
        peak = start
        running = True

        def poll() -> None:
            nonlocal peak
            while running:
                peak = max(peak, proc.memory_info().rss)
                time.sleep(interval)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        t0 = time.perf_counter()
        print(f"[mem] {label}: start rss={_gb(start):.2f} GB", flush=True)
        try:
            yield
        finally:
            running = False
            thread.join()
            end = proc.memory_info().rss
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            print(
                f"[mem] {label}: end rss={_gb(end):.2f} GB, "
                f"delta={_gb(end - start):+.2f} GB, "
                f"peak={_gb(peak):.2f} GB, "
                f"ru_maxrss={_gb(maxrss):.2f} GB, "
                f"time={time.perf_counter() - t0:.1f}s",
                flush=True,
            )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _gb(n_bytes: int) -> float:
    return n_bytes / 1024**3
