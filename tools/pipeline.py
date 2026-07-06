from __future__ import annotations

import copy
import datetime as dt
import gc
import hashlib
import io
import inspect
import json
import resource
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tools.data import POLARS_ENGINES, DataSource, Loader
from tools.precision import check_precision
from tools.registry import Registry
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
from tools.transform import Passthrough, Transform, load_transform, save_transform
from tools.score import Score, rmse, _NO_COMBINE


def evaluate_model(
    adapter: Any,
    model: Any,
    src: DataSource,
    score: Score,
    fold: Any = None,
    keep_predictions: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None]:
    if not getattr(adapter, "streaming", False):
        x, y_true, ctx = src.materialize()
        y_pred = adapter.predict(model, x)
        ctx["fold"] = fold
        return (
            score_arrays(score, y_true, y_pred, ctx),
            ctx,
            y_pred if keep_predictions else None,
        )
    loss, ctx, y_pred = score_stream(adapter, model, score, src, keep_predictions)
    ctx["fold"] = fold
    return loss, ctx, y_pred


def score_arrays(
    score: Score,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any],
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    return float(call_score(score, y_true, y_pred, ctx))


def score_stream(
    adapter: Any,
    model: Any,
    score: Score,
    src: DataSource,
    keep_predictions: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None]:
    state: Any = None
    ctx: dict[str, Any] = {"n": 0}
    pred_parts = [] if keep_predictions else None

    for x, y_true, batch_ctx in src.batches(adapter_batch_size(adapter)):
        y_pred = np.asarray(adapter.predict(model, x))
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )
        state = call_score(score, y_true, y_pred, batch_ctx, combine_with=state)
        merge_ctx(ctx, batch_ctx)
        if pred_parts is not None:
            pred_parts.append(y_pred)

    if state is None:
        raise ValueError("cannot score empty prediction stream")
    y_pred = np.concatenate(pred_parts) if pred_parts else None
    return float(state), ctx, y_pred


def call_score(
    score: Score,
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
    if has_varargs or (len(positional) >= 3 and positional[2].name != "combine_with"):
        args.append(ctx)
    elif "ctx" in sig.parameters:
        kwargs["ctx"] = ctx
    if combine_with is not _NO_COMBINE:
        if "combine_with" not in sig.parameters and not has_varkw:
            raise TypeError("streaming scores must accept a combine_with argument")
        kwargs["combine_with"] = combine_with
    return score(*args, **kwargs)


def adapter_batch_size(adapter: Any) -> int | None:
    return getattr(adapter, "batch_size", None)


def merge_ctx(total: dict[str, Any], batch: dict[str, Any]) -> None:
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
    for key, value in batch.items():
        if key not in {"n", "date", "dates", "nature", "natures"}:
            total[key] = value


def plot_train_val_loss(
    history: Any,
    *,
    trial: int | str | None = "best",
    fold: int | str | None = None,
    metric: str | None = None,
    ax: Any | None = None,
    axes: Any | None = None,
    max_cols: int = 2,
    title: str | None = None,
    score_direction: str | None = None,
    complete_trials: bool = True,
) -> Any:
    """Plot per-fit train/validation curves from Pipeline fit history.

    Accepts a ``Pipeline.train()`` result, a ``save_history()`` payload, raw
    ``pipeline.validation_history``, one validation record with ``fit_history``,
    or one record returned by ``pipeline.refit()``. By default, validation
    histories plot every fold from the best completed trial. Use ``trial=None``
    to plot all trials, ``fold=<n>`` to select one fold, or ``fold="best"`` to
    select the fold with the best available fit score. Pass
    ``complete_trials=False`` to allow automatic best-trial selection from
    pruned or otherwise incomplete trials.
    """

    def optional_float(value: Any) -> float:
        if value is None:
            return np.nan
        try:
            result = float(value)
        except (TypeError, ValueError):
            return np.nan
        return result if np.isfinite(result) else np.nan

    def optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def is_fit_record(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        nested = value.get("history")
        return isinstance(nested, Mapping) and isinstance(
            nested.get("train"), Mapping
        )

    def sequence_value(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )

    def columnar_records(data: Mapping[str, Any]) -> list[dict[str, Any]]:
        columns = {
            key: list(value) if sequence_value(value) else [value]
            for key, value in data.items()
        }
        n = max((len(value) for value in columns.values()), default=0)
        return [
            {
                key: value[i] if i < len(value) else None
                for key, value in columns.items()
            }
            for i in range(n)
        ]

    def best_record(
        items: Sequence[dict[str, Any]], direction: str
    ) -> dict[str, Any] | None:
        scored = []
        for item in items:
            score = np.nan
            for key in ("weighted_score", "val_score", "best_score"):
                score = optional_float(item.get(key))
                if np.isfinite(score):
                    break
            if not np.isfinite(score):
                nested = item.get("history", {})
                val_metrics = nested.get("val") if isinstance(nested, Mapping) else None
                train_metrics = (
                    nested.get("train") if isinstance(nested, Mapping) else None
                )
                for metrics in (val_metrics, train_metrics):
                    if not isinstance(metrics, Mapping):
                        continue
                    for values in metrics.values():
                        curve_values = np.asarray(
                            [optional_float(value) for value in values],
                            dtype=float,
                        )
                        if curve_values.size:
                            score = float(np.nanmin(curve_values))
                            break
                    if np.isfinite(score):
                        break
            if np.isfinite(score):
                scored.append((score, item))
        if not scored:
            return items[0] if items else None
        return (max if direction == "maximize" else min)(
            scored, key=lambda item: item[0]
        )[1]

    def best_choice(value: Any) -> bool:
        return isinstance(value, str) and value.lower() in {"best", "optimal"}

    def same_trial(left: Any, right: Any) -> bool:
        if left == right:
            return True
        left_int = optional_int(left)
        right_int = optional_int(right)
        if left_int is not None and right_int is not None:
            return left_int == right_int
        return str(left) == str(right)

    def expected_folds(
        items: Sequence[dict[str, Any]],
        grouped: Mapping[Any, Sequence[dict[str, Any]]],
    ) -> int | None:
        meta_folds = optional_int(meta.get("n_folds"))
        if meta_folds and meta_folds > 0:
            return meta_folds
        record_folds = [
            value
            for item in items
            if (value := optional_int(item.get("n_folds"))) is not None and value > 0
        ]
        if record_folds:
            return max(record_folds)
        group_sizes = []
        for group in grouped.values():
            folds = {item.get("fold") for item in group if item.get("fold") is not None}
            if folds:
                group_sizes.append(len(folds))
        return max(group_sizes, default=None)

    def is_complete_trial(items: Sequence[dict[str, Any]], n_folds: int | None) -> bool:
        if n_folds is None or n_folds <= 0:
            return True
        folds = {item.get("fold") for item in items if item.get("fold") is not None}
        if folds:
            return len(folds) >= n_folds
        return len(items) >= n_folds

    def curve(record: dict[str, Any], split: str, metric_name: str) -> np.ndarray:
        nested = record.get("history", {})
        metrics = nested.get(split, {}) if isinstance(nested, Mapping) else {}
        values = metrics.get(metric_name, []) if isinstance(metrics, Mapping) else []
        return np.asarray([optional_float(value) for value in values], dtype=float)

    meta: dict[str, Any] = {}
    data = history
    if is_fit_record(history):
        records = [dict(history)]
    else:
        if isinstance(history, Mapping):
            meta = dict(history)
            data = history.get("validation_history", history.get("fit_history"))
            if data is None:
                has_columns = any(sequence_value(value) for value in history.values())
                data = columnar_records(history) if has_columns else [history]
        if is_fit_record(data):
            records = [dict(data)]
        else:
            if hasattr(data, "to_dicts"):
                data = data.to_dicts()
            elif hasattr(data, "to_dict") and not isinstance(data, Mapping):
                try:
                    data = data.to_dict("records")
                except TypeError:
                    pass
            if is_fit_record(data):
                records = [dict(data)]
            else:
                if isinstance(data, Mapping):
                    has_columns = any(sequence_value(value) for value in data.values())
                    data = columnar_records(data) if has_columns else [data]
                records = []
                for item in data or []:
                    if not isinstance(item, Mapping):
                        continue
                    if is_fit_record(item):
                        records.append(dict(item))
                        continue
                    fit_history = item.get("fit_history")
                    if not is_fit_record(fit_history):
                        continue
                    record = dict(fit_history)
                    for key in (
                        "trial",
                        "fold",
                        "val_score",
                        "weighted_score",
                        "n_folds",
                        "n",
                        "dates",
                        "natures",
                        "params",
                    ):
                        if key in item and key not in record:
                            record[key] = item[key]
                    records.append(record)

    direction = (score_direction or meta.get("score_direction") or "minimize").lower()
    if best_choice(trial):
        chosen_trial = meta.get("best_trial")
        if chosen_trial is not None:
            records = [
                item for item in records if same_trial(item.get("trial"), chosen_trial)
            ]
        final_by_trial: dict[Any, dict[str, Any]] = {}
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for item in records:
            if item.get("trial") is not None:
                grouped.setdefault(item["trial"], []).append(item)
                final_by_trial[item["trial"]] = item
        if chosen_trial is None:
            candidates = grouped
            if complete_trials and grouped:
                n_folds = expected_folds(records, grouped)
                complete = {
                    trial_id: items
                    for trial_id, items in grouped.items()
                    if is_complete_trial(items, n_folds)
                }
                if complete:
                    candidates = complete
            best = best_record(
                [final_by_trial[trial_id] for trial_id in candidates],
                direction,
            )
            chosen_trial = best.get("trial") if best is not None else None
            if chosen_trial is not None:
                records = [
                    item
                    for item in records
                    if same_trial(item.get("trial"), chosen_trial)
                ]
    elif trial is not None:
        records = [item for item in records if same_trial(item.get("trial"), trial)]

    if best_choice(fold):
        best = best_record(records, direction)
        records = [best] if best is not None else []
    elif fold is not None:
        records = [item for item in records if item.get("fold") == fold]
    if not records:
        raise ValueError("no fit-history records match the requested filters")

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_train_val_loss() requires matplotlib.") from exc

    if ax is not None and axes is not None:
        raise ValueError("pass either ax or axes, not both")
    if ax is not None:
        if len(records) != 1:
            raise ValueError("ax can only be used when exactly one curve is selected")
        fig, flat_axes = ax.figure, np.asarray([ax], dtype=object)
    elif axes is not None:
        flat_axes = np.ravel(np.asarray(axes, dtype=object))
        if len(flat_axes) < len(records):
            raise ValueError(f"need at least {len(records)} axes, got {len(flat_axes)}")
        fig = flat_axes[0].figure
    else:
        if max_cols < 1:
            raise ValueError("max_cols must be positive")
        n_cols = min(max_cols, len(records))
        n_rows = int(np.ceil(len(records) / n_cols))
        fig, created_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.0 * n_cols, 3.8 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = np.ravel(created_axes)

    for record, item_ax in zip(records, flat_axes):
        nested = record.get("history", {})
        train_metrics = nested.get("train", {}) if isinstance(nested, Mapping) else {}
        val_metrics = nested.get("val", {}) if isinstance(nested, Mapping) else {}
        if metric is not None:
            metric_name = metric
        else:
            metric_name = None
            for name in val_metrics:
                if name in train_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                for name in train_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                for name in val_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                raise ValueError("fit history contains no train or validation metrics")
        train = curve(record, "train", metric_name)
        val = curve(record, "val", metric_name)
        if not train.size and not val.size:
            raise ValueError(f"metric {metric_name!r} has no train or val curve")
        if train.size:
            item_ax.plot(
                np.arange(1, len(train) + 1),
                train,
                color="#2563eb",
                linewidth=1.8,
                label=f"train {metric_name}",
            )
        if val.size:
            item_ax.plot(
                np.arange(1, len(val) + 1),
                val,
                color="#dc2626",
                linewidth=1.8,
                label=f"val {metric_name}",
            )
        best_iteration = record.get("best_iteration")
        if best_iteration is not None:
            try:
                item_ax.axvline(
                    int(best_iteration) + 1,
                    color="#f59e0b",
                    linestyle="--",
                    linewidth=1.0,
                    label="best",
                )
            except (TypeError, ValueError):
                pass

        parts = []
        if record.get("role") is not None:
            parts.append(str(record["role"]))
        if record.get("trial") is not None:
            parts.append(f"trial {record['trial']}")
        if record.get("fold") is not None:
            parts.append(f"fold {record['fold']}")
        item_title = " / ".join(parts) if parts else "fit"
        best_score = optional_float(record.get("best_score"))
        if np.isfinite(best_score):
            item_title += f" | best={best_score:.6g}"

        item_ax.set_title(item_title)
        item_ax.set_xlabel("round / epoch")
        item_ax.set_ylabel(metric_name)
        item_ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        item_ax.spines["top"].set_visible(False)
        item_ax.spines["right"].set_visible(False)
        item_ax.legend(frameon=False)

    for item_ax in flat_axes[len(records) :]:
        item_ax.set_visible(False)
    if title is not None:
        fig.suptitle(title)
    return fig


plot_validation_history = plot_train_val_loss


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
    precision: str = "float64"
    refit_val_dates: list[str] | None = None

    model: Any = field(default=None, init=False)
    best_params: dict[str, Any] | None = field(default=None, init=False)
    study: Any = field(default=None, init=False)
    fitted_transform: Transform | None = field(default=None, init=False)
    validation_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    refit_history: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.score_direction = self.score_direction.lower()
        if self.score_direction not in {"minimize", "maximize"}:
            raise ValueError("score_direction must be 'minimize' or 'maximize'")
        self.polars_engine = self.polars_engine.lower()
        if self.polars_engine not in POLARS_ENGINES:
            raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
        self.precision = check_precision(self.precision)
        self.train_filters = tuple(self.train_filters)
        self.val_filters = tuple(self.val_filters)
        self.test_filters = tuple(self.test_filters)
        if self.refit_val_dates is not None:
            self.refit_val_dates = (
                [self.refit_val_dates]
                if isinstance(self.refit_val_dates, str)
                else list(self.refit_val_dates)
            )
        self._transform_cache: dict[tuple[Any, ...], Transform] = {}
        self._array_cache: dict[
            tuple[Any, ...], tuple[np.ndarray, np.ndarray, dict[str, Any]]
        ] = {}

    def train(
        self,
        verbose: int = 0,
        memory_log: bool = False,
        memory_interval: float = 0.05,
        no_refit: bool = False,
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
                "precision": self.precision,
                "no_refit": no_refit,
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
                        model = self._fit_model(
                            model,
                            train_src,
                            val_src,
                            trial,
                            fit_context={
                                "role": "cv",
                                "trial": trial.number,
                                "fold": fold,
                                "n_folds": len(folds),
                                "train_dates": list(train_dates),
                                "val_dates": list(val_dates),
                                "val_score": self.val_score,
                                "score_direction": self.score_direction,
                                "score_name": _callable_name(self.val_score),
                            },
                        )
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
                        "n_folds": len(folds),
                        "val_score": loss,
                        "weighted_score": running,
                        "n": int(ctx["n"]),
                        "dates": ctx.get("dates"),
                        "natures": ctx.get("natures"),
                        "params": params,
                    }
                    fit_history = getattr(self.adapter, "last_fit_history", None)
                    if fit_history is not None:
                        record["fit_history"] = fit_history
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
                    if fold + 1 < len(folds) and trial.should_prune():
                        raise optuna.TrialPruned()
                return weighted_mean(scores, sizes, self.fold_weighting)

            self.study.optimize(objective, n_trials=self.n_trials)
            if verbose > 0:
                action = (
                    "Skipping final refit."
                    if no_refit
                    else "Refitting with best params."
                )
                print(
                    f"======== optimization finished, best params extracted. {action}"
                )
            self.best_params = dict(
                self.study.best_trial.user_attrs.get("params", self.study.best_params)
            )
            self.model = None
            self.fitted_transform = None
            self.refit_history = None
            if not no_refit:
                self.refit(
                    dates=self._all_train_dates(),
                    val_dates=self.refit_val_dates,
                    params=self.best_params,
                    memory_log=memory_log,
                    memory_interval=memory_interval,
                )
            print("======== search done." if no_refit else "======== training done.")
            return {
                "best_params": self.best_params,
                "best_score": float(self.study.best_value),
                "best_trial": int(self.study.best_trial.number),
                "n_trials": len(self.study.trials),
                "n_folds": len(folds),
                "refit": not no_refit,
                "validation_history": self.validation_history,
                "refit_history": self.refit_history,
            }
        finally:
            self.tracker.finish()

    def test(
        self, score: Score = rmse, keep_predictions: bool = True
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() and refit() before test()")
        transform = self.fitted_transform or self._fit_transform(
            self._all_train_dates()
        )
        src = self._src(self.test_dates, self.test_filters, transform, "test")
        loss, ctx, y_pred = self._evaluate(
            self.model, src, score, "test", keep_predictions=keep_predictions
        )
        return {"test_score": loss, "n": int(ctx["n"]), "ctx": ctx, "y_pred": y_pred}

    def get_model(self) -> Any:
        return self.model

    def save_model(
        self, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before save_model()")
        return self.adapter.save_model(self.model, path, filename=filename)

    def load_model(
        self,
        path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        self.model = self.adapter.load_model(path, meta)
        return self.model

    def save_history(self, path: str | Path) -> dict[str, Any]:
        payload = self._history_payload()
        _write_json(Path(path), payload)
        return payload

    def save_pipeline(self, path: str | Path) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before save_pipeline()")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        run_hash, stamp, run_dir = self._new_run_dir(root)
        registry = Registry(root / "registry.jsonl")

        model_meta = self.save_model(root / "model", filename=f"model_{run_hash}")
        model_meta["path"] = f"model/{model_meta['artifact']}"

        transform_path = root / "transform" / f"transform_{run_hash}.pkl"
        transform_meta = save_transform(self.fitted_transform or self.transform, transform_path)
        transform_meta["path"] = f"transform/{transform_path.name}"

        self.save_history(run_dir / "history.json")

        filters = {
            role: self._register_filters(registry, exprs, f"{role}_filter", run_hash)
            for role, exprs in (
                ("train", self.train_filters),
                ("val", self.val_filters),
                ("test", self.test_filters),
            )
        }
        features = self._register_features(registry, run_hash)

        manifest = {
            "version": 2,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "run": {"stamp": stamp, "hash": run_hash},
            "target": self.target,
            "features": list(self.features),
            "feature_registry": features,
            "rolling_dates": self.rolling_dates,
            "test_dates": self.test_dates,
            "best_params": self.best_params,
            "score_direction": self.score_direction,
            "val_score": _callable_name(self.val_score),
            "sampler": self.sampler,
            "n_trials": self.n_trials,
            "polars_engine": self.polars_engine,
            "refit_val_dates": self.refit_val_dates,
            "adapter": _object_info(self.adapter),
            "data_loader": _object_info(self.data_loader),
            "model": model_meta,
            "transform": transform_meta,
            "filters": filters,
            "registry": {"path": "registry.jsonl"},
            "history": {"path": "history.json"},
        }
        _write_json(run_dir / "pipeline.json", manifest)
        return manifest

    def load_pipeline(self, path: str | Path) -> dict[str, Any]:
        run_dir = Path(path)
        if not (run_dir / "pipeline.json").exists():
            candidates = sorted(
                p for p in run_dir.glob("pipeline_*") if (p / "pipeline.json").exists()
            )
            if not candidates:
                raise FileNotFoundError(f"no pipeline manifest found under {run_dir}")
            run_dir = candidates[-1]
        manifest = _read_json(run_dir / "pipeline.json")
        run_info = manifest.get("run")
        root = run_dir.parent if run_info is not None else run_dir

        self.target = manifest.get("target", self.target)
        self.features = list(manifest.get("features", self.features))
        self.rolling_dates = [
            list(x) for x in manifest.get("rolling_dates", self.rolling_dates)
        ]
        self.test_dates = list(manifest.get("test_dates", self.test_dates))
        self.refit_val_dates = manifest.get("refit_val_dates", self.refit_val_dates)
        self.best_params = manifest.get("best_params")
        self.score_direction = manifest.get("score_direction", self.score_direction)
        self.polars_engine = manifest.get("polars_engine", self.polars_engine)

        history_path = manifest.get("history", {}).get("path", "history.json")
        history = _read_json(run_dir / history_path, {})
        self.validation_history = list(history.get("validation_history", []))
        self.refit_history = history.get("refit_history")

        transform_meta = manifest.get("transform")
        if transform_meta:
            self.fitted_transform = load_transform(
                root / transform_meta["path"], transform_meta
            )
        model_meta = manifest.get("model")
        if model_meta:
            self.load_model(root / model_meta["path"], model_meta)

        registry_meta = manifest.get("registry")
        registry = Registry(root / registry_meta["path"]) if registry_meta else None
        filters = manifest.get("filters", {})
        self.train_filters = tuple(_exprs_from_refs(filters.get("train", []), registry))
        self.val_filters = tuple(_exprs_from_refs(filters.get("val", []), registry))
        self.test_filters = tuple(_exprs_from_refs(filters.get("test", []), registry))
        return manifest

    def _new_run_dir(self, root: Path) -> tuple[str, str, Path]:
        while True:
            now = dt.datetime.now(dt.UTC)
            run_hash = hashlib.sha256(
                now.isoformat(timespec="milliseconds").encode()
            ).hexdigest()[:8]
            stamp = now.strftime("%Y%m%dT%H%M")
            run_dir = root / f"pipeline_{stamp}_{run_hash}"
            try:
                run_dir.mkdir(parents=True)
            except FileExistsError:
                continue
            return run_hash, stamp, run_dir

    def _register_filters(
        self,
        registry: Registry,
        exprs: Sequence[pl.Expr],
        prefix: str,
        run_hash: str,
    ) -> list[dict[str, Any]]:
        return [
            _register_expr(registry, "filter", f"{prefix}_{i}", expr, run_hash)
            for i, expr in enumerate(exprs)
        ]

    def _register_features(
        self, registry: Registry, run_hash: str
    ) -> dict[str, dict[str, Any]]:
        exprs = _feature_exprs(self.data_loader)
        refs: dict[str, dict[str, Any]] = {}
        for name in self.features:
            expr = exprs.get(name)
            if expr is None:
                continue
            refs[name] = _register_expr(registry, "feature", name, expr, run_hash)
        return refs

    def refit(
        self,
        dates: Sequence[str] | None = None,
        val_dates: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> Any:
        if params is None:
            if self.best_params is None:
                raise RuntimeError("call train() before refit() or pass params")
            params = self.best_params
        if dates is None:
            train_dates = self._all_train_dates()
        elif isinstance(dates, str):
            train_dates = [dates]
        else:
            train_dates = list(dates)
        if val_dates is None:
            final_val_dates = None
        elif isinstance(val_dates, str):
            final_val_dates = [val_dates]
        else:
            final_val_dates = list(val_dates)
        if final_val_dates:
            val_set = set(final_val_dates)
            train_dates = [date for date in train_dates if date not in val_set]
            if not train_dates:
                raise ValueError("refit validation dates leave no training dates")
        with self._memory_log(
            f"refit _fit_transform train_dates={train_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            self.fitted_transform = self._fit_transform(train_dates)
        train_src = self._src(
            train_dates, self.train_filters, self.fitted_transform, "final_train"
        )
        val_src = (
            self._src(
                final_val_dates,
                self.val_filters,
                self.fitted_transform,
                "final_val",
            )
            if final_val_dates
            else None
        )
        model = self.adapter.build(params)
        with self._memory_log(
            f"refit _fit_model train_dates={train_dates} val_dates={final_val_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            self.model = self._fit_model(
                model,
                train_src,
                val_src,
                None,
                fit_context={
                    "role": "refit",
                    "trial": None,
                    "fold": None,
                    "train_dates": list(train_dates),
                    "val_dates": list(final_val_dates) if final_val_dates else None,
                    "val_score": self.val_score,
                    "score_direction": self.score_direction,
                    "score_name": _callable_name(self.val_score),
                },
            )
        self.refit_history = getattr(self.adapter, "last_fit_history", None)
        return self.refit_history

    def _refit(
        self,
        params: dict[str, Any],
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> Any:
        self.refit(
            dates=self._all_train_dates(),
            params=params,
            memory_log=memory_log,
            memory_interval=memory_interval,
        )
        return self.model

    def _fit_transform(self, dates: Sequence[str]) -> Transform:
        key = (tuple(dates), self.polars_engine, self.precision)
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
        key = (role, tuple(dates), id(transform), self.polars_engine, self.precision)
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
            precision=self.precision,
        )

    def _fit_model(
        self,
        model: Any,
        train: DataSource,
        val: DataSource | None,
        trial: Any | None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        sig = inspect.signature(self.adapter.fit)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: dict[str, Any] = {}
        accepts_trial = "trial" in sig.parameters or accepts_kwargs
        if accepts_trial:
            kwargs["trial"] = trial
        if "fit_context" in sig.parameters or accepts_kwargs:
            kwargs["fit_context"] = fit_context
        return self.adapter.fit(model, train, val, self.tracker, **kwargs)

    def _evaluate(
        self,
        model: Any,
        src: DataSource,
        score: Callable[..., float],
        fold: Any,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        evaluate = getattr(self.adapter, "evaluate", None)
        if evaluate is None:
            return evaluate_model(
                self.adapter, model, src, score, fold, keep_predictions
            )
        return evaluate(model, src, score, fold=fold, keep_predictions=keep_predictions)

    def _all_train_dates(self) -> list[str]:
        return [date for chunk in self.rolling_dates for date in chunk]

    def _history_payload(self) -> dict[str, Any]:
        best_score = None
        best_trial = None
        n_trials = 0
        if self.study is not None:
            n_trials = len(getattr(self.study, "trials", []))
            try:
                best_score = float(self.study.best_value)
            except (ValueError, AttributeError):
                pass
            try:
                best_trial = int(self.study.best_trial.number)
            except (ValueError, AttributeError):
                pass
        return {
            "best_params": self.best_params,
            "best_score": best_score,
            "best_trial": best_trial,
            "score_direction": self.score_direction,
            "val_score": _callable_name(self.val_score),
            "n_trials": n_trials,
            "n_folds": max(0, len(self.rolling_dates) - 1),
            "refit_val_dates": self.refit_val_dates,
            "validation_history": self.validation_history,
            "refit_history": self.refit_history,
        }

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


def _register_expr(
    registry: Registry, kind: str, name: str, expr: pl.Expr, run_hash: str
) -> dict[str, Any]:
    blob = expr.meta.serialize(format="json")
    fingerprint = hashlib.sha256(blob.encode()).hexdigest()
    entry = registry.register(
        kind,
        name,
        fingerprint,
        {
            "expr": blob,
            "repr": str(expr),
            "roots": expr.meta.root_names(),
            "format": "json",
            "polars_version": pl.__version__,
        },
        run=run_hash,
    )
    return {"name": name, "version": entry["version"], "fingerprint": fingerprint}


def _exprs_from_refs(
    configs: Sequence[dict[str, Any]], registry: Registry | None
) -> list[pl.Expr]:
    exprs = []
    for cfg in configs:
        if "expr" in cfg:
            exprs.append(
                pl.Expr.deserialize(
                    io.StringIO(cfg["expr"]), format=cfg.get("format", "json")
                )
            )
            continue
        if registry is None:
            raise RuntimeError(
                f"cannot resolve filter {cfg.get('name')!r}: no registry available"
            )
        entry = registry.resolve("filter", cfg["name"], cfg["version"])
        if entry["fingerprint"] != cfg.get("fingerprint", entry["fingerprint"]):
            raise ValueError(
                f"registry fingerprint mismatch for filter {cfg['name']!r} "
                f"v{cfg['version']}"
            )
        exprs.append(
            pl.Expr.deserialize(
                io.StringIO(entry["expr"]), format=entry.get("format", "json")
            )
        )
    return exprs


def _feature_exprs(loader: Any) -> Mapping[str, pl.Expr]:
    seen: set[int] = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        exprs = getattr(loader, "feature_exprs", None)
        if exprs:
            return exprs
        loader = getattr(loader, "loader", None)
    return {}


def _read_json(path: Path, default: Any | None = None) -> Any:
    if default is not None and not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(_json_ready(obj), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _callable_name(fn: Any) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _object_info(obj: Any) -> dict[str, Any]:
    cls = type(obj)
    return {"class": cls.__name__, "module": cls.__module__, "repr": repr(obj)}
