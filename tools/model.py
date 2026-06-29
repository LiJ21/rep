from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from tools.data import DataSource
    from tools.track import Tracker


@runtime_checkable
class ModelAdapter(Protocol):
    streaming: bool

    def build(self, params: dict[str, Any]) -> Any: ...

    def fit(
        self,
        model: Any,
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any: ...

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray: ...


@dataclass
class DummyAdapter:
    mode: str = "mean"
    streaming: bool = False

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"params": dict(params)}

    def fit(
        self,
        model: dict[str, Any],
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        x, y, _ = train.materialize()
        if self.mode == "linear" and x.size:
            design = np.c_[np.ones(len(x)), x]
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            model.update({"intercept": coef[0], "coef": coef[1:]})
        else:
            model["mean"] = float(np.mean(y)) if len(y) else 0.0
        return model

    def predict(self, model: dict[str, Any], x: np.ndarray) -> np.ndarray:
        if "coef" in model:
            return np.asarray(model["intercept"] + x @ model["coef"])
        return np.full(len(x), model.get("mean", 0.0), dtype=float)


@dataclass
class XGBoostAdapter:
    num_boost_round: int = 100
    early_stopping_rounds: int | None = None
    batch_size: int | None = 200_000
    streaming: bool = True
    callbacks: list[Any] = field(default_factory=list)
    pruning_metric: str | None = None

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def fit(
        self,
        model: dict[str, Any],
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoostAdapter requires xgboost.") from exc

        dtrain = self._dmatrix(xgb, train)
        evals = []
        if val is not None:
            evals.append((self._dmatrix(xgb, val, ref=dtrain), "val"))
        callbacks = list(self.callbacks)
        if trial is not None and self.pruning_metric:
            try:
                from optuna.integration import XGBoostPruningCallback

                callbacks.append(XGBoostPruningCallback(trial, self.pruning_metric))
            except ImportError:
                warnings.warn(
                    "XGBoost pruning requested, but optuna-integration is not installed; "
                    "continuing without pruning callback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        history: dict[str, Any] = {}
        booster = xgb.train(
            model,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            evals_result=history,
            early_stopping_rounds=self.early_stopping_rounds if evals else None,
            callbacks=callbacks,
            verbose_eval=False,
        )
        for name, metrics in history.items():
            for metric, values in metrics.items():
                if values:
                    tracker.log({f"xgb/{name}_{metric}": values[-1]})
        if trial is not None:
            self._record_trial_fit(trial, booster, fit_context)
        return booster

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        return np.asarray(model.inplace_predict(x))

    def _dmatrix(self, xgb: Any, src: "DataSource", ref: Any | None = None) -> Any:
        if not self.streaming:
            x, y, _ = src.materialize()
            return xgb.DMatrix(x, label=y)

        batch_size = self.batch_size

        class BatchIter(xgb.DataIter):
            def __init__(self):
                super().__init__()
                self._iter = None

            def reset(self):
                self._iter = iter(src.batches(batch_size))

            def next(self, input_data):
                try:
                    x, y, _ = next(self._iter)
                except StopIteration:
                    return 0
                input_data(data=x, label=y)
                return 1

        return xgb.QuantileDMatrix(BatchIter(), ref=ref)

    def _record_trial_fit(
        self,
        trial: Any,
        booster: Any,
        fit_context: dict[str, Any] | None,
    ) -> None:
        record = self._fit_record(booster, fit_context)
        records = list(trial.user_attrs.get("xgb_fits", []))
        records.append(record)
        trial.set_user_attr("xgb_fits", records)

        cv_rounds = [
            int(item["best_num_boost_round"])
            for item in records
            if item.get("role") == "cv" and item.get("best_num_boost_round") is not None
        ]
        if cv_rounds:
            trial.set_user_attr("xgb_cv_best_num_boost_rounds", cv_rounds)
            trial.set_user_attr(
                "xgb_cv_best_num_boost_round_median",
                float(np.median(cv_rounds)),
            )
            trial.set_user_attr("xgb_cv_best_num_boost_round_max", max(cv_rounds))

    def _fit_record(
        self,
        booster: Any,
        fit_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ctx = fit_context or {}
        best_iteration = _as_optional_int(_safe_attr(booster, "best_iteration"))
        best_score = _as_optional_float(_safe_attr(booster, "best_score"))
        record: dict[str, Any] = {
            "role": ctx.get("role"),
            "fold": ctx.get("fold"),
            "train_dates": ctx.get("train_dates"),
            "val_dates": ctx.get("val_dates"),
            "best_iteration": best_iteration,
            "best_num_boost_round": (
                best_iteration + 1 if best_iteration is not None else None
            ),
            "best_score": best_score,
            "num_boosted_rounds": _num_boosted_rounds(booster),
        }
        return record


def _safe_attr(obj: Any, name: str) -> Any | None:
    try:
        return getattr(obj, name)
    except (AttributeError, ValueError):
        return None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num_boosted_rounds(booster: Any) -> int | None:
    try:
        return int(booster.num_boosted_rounds())
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass
class LarsAdapter:
    alpha: float = 1.0
    method: str = "lasso"
    fit_intercept: bool = True
    max_iter: int = 500
    alpha_min: float = 0.0
    eps: float = float(np.finfo(float).eps)
    positive: bool = False
    batch_size: int | None = 200_000
    max_features: int | None = None
    stats_backend: str = "cpu"
    stats_dtype: Any = np.float64
    streaming: bool = True
    cache_path: bool = True
    _stats_cache: dict[Any, "_LinearStats"] = field(
        default_factory=dict, init=False, repr=False
    )
    _path_cache: dict[Any, tuple[np.ndarray, list[int], np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, params: dict[str, Any]) -> "LarsPathModel":
        cfg = {
            "alpha": self.alpha,
            "method": self.method,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "alpha_min": self.alpha_min,
            "eps": self.eps,
            "positive": self.positive,
            **params,
        }
        known = {
            "alpha",
            "method",
            "fit_intercept",
            "max_iter",
            "alpha_min",
            "eps",
            "positive",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unsupported LarsAdapter params: {sorted(unknown)}")
        return LarsPathModel(
            alpha=float(cfg["alpha"]),
            method=str(cfg["method"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            max_iter=int(cfg["max_iter"]),
            alpha_min=float(cfg["alpha_min"]),
            eps=float(cfg["eps"]),
            positive=bool(cfg["positive"]),
            params=dict(params),
        )

    def fit(
        self,
        model: "LarsPathModel",
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> "LarsPathModel":
        try:
            from sklearn.linear_model import lars_path_gram
        except ImportError as exc:
            raise ImportError("LarsAdapter requires scikit-learn.") from exc

        stats = self._stats(train)
        gram, xy, x_mean, y_mean = stats.centered(model.fit_intercept)
        key = self._path_key(train, model)
        cached = self._path_cache.get(key) if key is not None else None
        if cached is None:
            alphas, active, coefs_path = lars_path_gram(
                xy,
                gram,
                n_samples=stats.n,
                method=model.method,
                max_iter=model.max_iter,
                alpha_min=model.alpha_min,
                eps=model.eps,
                positive=model.positive,
                return_path=True,
            )
            cached = (np.asarray(alphas), [int(i) for i in active], np.asarray(coefs_path))
            if key is not None:
                self._path_cache[key] = cached
        model.alphas, model.active, model.coefs_path = cached
        model.n_samples = stats.n
        model.x_mean = x_mean
        model.y_mean = y_mean
        model.coef = self._coef_at(model.alphas, model.coefs_path, model.alpha)
        model.intercept = float(y_mean - x_mean @ model.coef) if model.fit_intercept else 0.0
        if tracker is not None:
            tracker.log(
                {
                    "lars/alpha": model.alpha,
                    "lars/path_len": len(model.alphas),
                    "lars/n": stats.n,
                    "lars/n_features": stats.n_features,
                }
            )
        return model

    def predict(self, model: "LarsPathModel", x: np.ndarray) -> np.ndarray:
        return model.predict(x)

    def path(self, model: "LarsPathModel") -> dict[str, Any]:
        if model.alphas is None or model.coefs_path is None:
            raise RuntimeError("LARS path is not available before fit()")
        return {
            "alphas": model.alphas.copy(),
            "active": list(model.active),
            "coefs_path": model.coefs_path.copy(),
        }

    def _stats(self, src: "DataSource") -> "_LinearStats":
        key = self._stats_key(src)
        if key is not None and key in self._stats_cache:
            return self._stats_cache[key]

        stats: _LinearStats | None = None
        for x, y, _ in src.batches(self.batch_size):
            x = np.asarray(x)
            y = np.asarray(y).reshape(-1)
            if len(y) == 0:
                continue
            if stats is None:
                if self.max_features is not None and x.shape[1] > self.max_features:
                    raise MemoryError(
                        f"LarsAdapter got {x.shape[1]} features; max_features={self.max_features}"
                    )
                stats = _LinearStats.empty(x.shape[1], self._stats_xp(), self.stats_dtype)
            stats.add(x, y)

        if stats is None or stats.n == 0:
            raise ValueError("cannot fit LarsAdapter on empty data")
        if key is not None:
            self._stats_cache[key] = stats
        return stats

    def _stats_key(self, src: "DataSource") -> Any | None:
        key = getattr(src, "cache_key", None)
        if key is None:
            return None
        return (
            key,
            tuple(src.features),
            src.target,
            self.stats_backend,
            str(np.dtype(self.stats_dtype)),
        )

    def _path_key(self, src: "DataSource", model: "LarsPathModel") -> Any | None:
        key = self._stats_key(src)
        if not self.cache_path or key is None:
            return None
        return (
            key,
            model.method,
            model.fit_intercept,
            model.max_iter,
            model.alpha_min,
            model.eps,
            model.positive,
        )

    def _stats_xp(self) -> Any:
        name = self.stats_backend.lower()
        if name in {"cpu", "numpy"}:
            return np
        if name in {"cupy", "gpu"}:
            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError(
                    "LarsAdapter stats_backend='cupy' requires CuPy; for CUDA 12 install cupy-cuda12x, not cupy."
                ) from exc
            return cp
        raise ValueError("stats_backend must be one of: 'cpu', 'numpy', 'cupy', 'gpu'")

    @staticmethod
    def _coef_at(alphas: np.ndarray, coefs_path: np.ndarray, alpha: float) -> np.ndarray:
        if len(alphas) == 1:
            return coefs_path[:, 0].copy()
        if alpha < alphas[-1]:
            warnings.warn(
                f"alpha={alpha} is below computed path minimum {alphas[-1]}; using path endpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        xp = alphas[::-1]
        return np.array([np.interp(alpha, xp, row[::-1]) for row in coefs_path], dtype=float)


@dataclass
class LarsPathModel:
    alpha: float
    method: str = "lasso"
    fit_intercept: bool = True
    max_iter: int = 500
    alpha_min: float = 0.0
    eps: float = float(np.finfo(float).eps)
    positive: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    alphas: np.ndarray | None = None
    active: list[int] = field(default_factory=list)
    coefs_path: np.ndarray | None = None
    n_samples: int = 0
    x_mean: np.ndarray | None = None
    y_mean: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("LarsPathModel is not fitted")
        return np.asarray(x, dtype=float) @ self.coef + self.intercept


@dataclass
class RidgeAdapter:
    alpha: float = 1.0
    fit_intercept: bool = True
    batch_size: int | None = 200_000
    max_features: int | None = None
    stats_backend: str = "cpu"
    stats_dtype: Any = np.float64
    streaming: bool = True
    cache_stats: bool = True
    _stats_cache: dict[Any, "_LinearStats"] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, params: dict[str, Any]) -> "RidgeModel":
        cfg = {"alpha": self.alpha, "fit_intercept": self.fit_intercept, **params}
        unknown = set(cfg) - {"alpha", "fit_intercept"}
        if unknown:
            raise ValueError(f"unsupported RidgeAdapter params: {sorted(unknown)}")
        return RidgeModel(
            alpha=float(cfg["alpha"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            params=dict(params),
        )

    def fit(
        self,
        model: "RidgeModel",
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> "RidgeModel":
        stats = self._stats(train)
        gram, xy, x_mean, y_mean = stats.centered(model.fit_intercept)
        reg = float(model.alpha) * np.eye(stats.n_features, dtype=gram.dtype)
        model.coef = _solve_linear(gram + reg, xy)
        model.intercept = (
            float(y_mean - x_mean @ model.coef) if model.fit_intercept else 0.0
        )
        model.n_samples = stats.n
        model.x_mean = x_mean
        model.y_mean = y_mean
        if tracker is not None:
            tracker.log(
                {
                    "ridge/alpha": model.alpha,
                    "ridge/n": stats.n,
                    "ridge/n_features": stats.n_features,
                }
            )
        return model

    def predict(self, model: "RidgeModel", x: np.ndarray) -> np.ndarray:
        return model.predict(x)

    def _stats(self, src: "DataSource") -> "_LinearStats":
        key = self._stats_key(src)
        if key is not None and key in self._stats_cache:
            return self._stats_cache[key]

        stats: _LinearStats | None = None
        for x, y, _ in src.batches(self.batch_size):
            x = np.asarray(x)
            y = np.asarray(y).reshape(-1)
            if len(y) == 0:
                continue
            if stats is None:
                if self.max_features is not None and x.shape[1] > self.max_features:
                    raise MemoryError(
                        f"RidgeAdapter got {x.shape[1]} features; max_features={self.max_features}"
                    )
                stats = _LinearStats.empty(x.shape[1], self._stats_xp(), self.stats_dtype)
            stats.add(x, y)

        if stats is None or stats.n == 0:
            raise ValueError("cannot fit RidgeAdapter on empty data")
        if key is not None:
            self._stats_cache[key] = stats
        return stats

    def _stats_key(self, src: "DataSource") -> Any | None:
        if not self.cache_stats:
            return None
        key = getattr(src, "cache_key", None)
        if key is None:
            return None
        return (
            key,
            tuple(src.features),
            src.target,
            self.stats_backend,
            str(np.dtype(self.stats_dtype)),
        )

    def _stats_xp(self) -> Any:
        name = self.stats_backend.lower()
        if name in {"cpu", "numpy"}:
            return np
        if name in {"cupy", "gpu"}:
            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError(
                    "RidgeAdapter stats_backend='cupy' requires CuPy; for CUDA 12 install cupy-cuda12x, not cupy."
                ) from exc
            return cp
        raise ValueError("stats_backend must be one of: 'cpu', 'numpy', 'cupy', 'gpu'")


@dataclass
class RidgeModel:
    alpha: float
    fit_intercept: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    n_samples: int = 0
    x_mean: np.ndarray | None = None
    y_mean: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("RidgeModel is not fitted")
        return np.asarray(x, dtype=float) @ self.coef + self.intercept


def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


@dataclass
class _LinearStats:
    xp: Any
    dtype: Any
    xtx: Any
    xty: Any
    x_sum: Any
    y_sum: float = 0.0
    n: int = 0

    @classmethod
    def empty(
        cls, n_features: int, xp: Any = np, dtype: Any = np.float64
    ) -> "_LinearStats":
        dtype = xp.dtype(dtype)
        return cls(
            xp=xp,
            dtype=dtype,
            xtx=xp.zeros((n_features, n_features), dtype=dtype),
            xty=xp.zeros(n_features, dtype=dtype),
            x_sum=xp.zeros(n_features, dtype=dtype),
        )

    @property
    def n_features(self) -> int:
        return int(self.x_sum.size)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[0] != len(y):
            raise ValueError(f"batch length mismatch: X={x.shape[0]} y={len(y)}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"feature count changed: expected {self.n_features}, got {x.shape[1]}"
            )
        x = self.xp.asarray(x, dtype=self.dtype)
        y = self.xp.asarray(y, dtype=self.dtype).reshape(-1)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.x_sum += x.sum(axis=0)
        self.y_sum += float(self._cpu(y.sum()).item())
        self.n += len(y)

    def centered(self, fit_intercept: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not fit_intercept:
            return self._cpu(self.xtx), self._cpu(self.xty), np.zeros(self.n_features), 0.0
        x_mean = self.x_sum / self.n
        y_mean = self.y_sum / self.n
        gram = self.xtx - self.xp.outer(self.x_sum, self.x_sum) / self.n
        xy = self.xty - self.x_sum * y_mean
        return self._cpu(gram), self._cpu(xy), self._cpu(x_mean), float(y_mean)

    def _cpu(self, x: Any) -> np.ndarray:
        if self.xp is np:
            return np.asarray(x).copy()
        return self.xp.asnumpy(x)


@dataclass
class TorchAdapter:
    module_builder: Callable[[dict[str, Any]], Any]
    loss_fn: Any | None = None
    optimizer_builder: Callable[[Any, dict[str, Any]], Any] | None = None
    epochs: int = 1
    batch_size: int | None = 8192
    device: str | None = None
    distributed: bool = False
    streaming: bool = True

    def build(self, params: dict[str, Any]) -> Any:
        model = self.module_builder(params)
        setattr(model, "_pipeline_params", dict(params))
        return model

    def fit(
        self,
        model: Any,
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        import torch

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        rank, world_size = self._rank_world(torch)
        model_params = getattr(model, "_pipeline_params", {})
        model = model.to(device)
        if self.distributed and world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
        loss_fn = self.loss_fn or torch.nn.MSELoss()
        optimizer = self._optimizer(torch, model, model_params)

        for epoch in range(self.epochs):
            model.train()
            losses = []
            for xb, yb in self._loader(torch, train, rank, world_size):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb.float())
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = {"torch/train_loss": float(np.mean(losses)) if losses else 0.0}
            if val is not None:
                metrics["torch/val_loss"] = self._eval_loss(torch, model, val, loss_fn, device)
            tracker.log(metrics, step=epoch)
        return model

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        import torch

        device = next(model.parameters()).device
        model.eval()
        with torch.inference_mode():
            xb = torch.as_tensor(x, dtype=torch.float32, device=device)
            return model(xb).detach().cpu().numpy().reshape(-1)

    def _loader(self, torch: Any, src: "DataSource", rank: int, world_size: int):
        batch_size = self.batch_size

        class SourceDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for i, (x, y, _) in enumerate(src.batches(batch_size)):
                    if i % world_size == rank:
                        yield torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y)

        return torch.utils.data.DataLoader(SourceDataset(), batch_size=None)

    def _eval_loss(self, torch: Any, model: Any, src: "DataSource", loss_fn: Any, device: Any) -> float:
        losses = []
        model.eval()
        with torch.inference_mode():
            for xb, yb in self._loader(torch, src, 0, 1):
                pred = model(xb.to(device)).squeeze(-1)
                losses.append(float(loss_fn(pred, yb.to(device).float()).cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _optimizer(self, torch: Any, model: Any, params: dict[str, Any]) -> Any:
        if self.optimizer_builder is not None:
            return self.optimizer_builder(model.parameters(), params)
        return torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))

    def _rank_world(self, torch: Any) -> tuple[int, int]:
        if self.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1
