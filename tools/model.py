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
    ) -> Any: ...

    def predict(self, model: Any, src: "DataSource") -> np.ndarray: ...


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
    ) -> dict[str, Any]:
        x, y, _ = train.materialize()
        if self.mode == "linear" and x.size:
            design = np.c_[np.ones(len(x)), x]
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            model.update({"intercept": coef[0], "coef": coef[1:]})
        else:
            model["mean"] = float(np.mean(y)) if len(y) else 0.0
        return model

    def predict(self, model: dict[str, Any], src: "DataSource") -> np.ndarray:
        x, y, _ = src.materialize()
        if "coef" in model:
            return np.asarray(model["intercept"] + x @ model["coef"])
        return np.full(len(y), model.get("mean", 0.0), dtype=float)


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
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=callbacks,
            verbose_eval=False,
        )
        for name, metrics in history.items():
            for metric, values in metrics.items():
                if values:
                    tracker.log({f"xgb/{name}_{metric}": values[-1]})
        return booster

    def predict(self, model: Any, src: "DataSource") -> np.ndarray:
        preds = []
        for x, _, _ in src.batches(self.batch_size):
            preds.append(np.asarray(model.inplace_predict(x)))
        return np.concatenate(preds) if preds else np.array([], dtype=float)

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


@dataclass
class LarsAdapter:
    kind: str = "lasso_lars"
    streaming: bool = False

    def build(self, params: dict[str, Any]) -> Any:
        try:
            from sklearn import linear_model
        except ImportError as exc:
            raise ImportError("LarsAdapter requires scikit-learn.") from exc
        cls = linear_model.Lars if self.kind == "lars" else linear_model.LassoLars
        return cls(**params)

    def fit(
        self,
        model: Any,
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
    ) -> Any:
        x, y, _ = train.materialize()
        return model.fit(x, y)

    def predict(self, model: Any, src: "DataSource") -> np.ndarray:
        x, _, _ = src.materialize()
        return np.asarray(model.predict(x))


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

    def predict(self, model: Any, src: "DataSource") -> np.ndarray:
        import torch

        device = next(model.parameters()).device
        model.eval()
        preds = []
        with torch.inference_mode():
            for xb, _ in self._loader(torch, src, 0, 1):
                preds.append(model(xb.to(device)).detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds) if preds else np.array([], dtype=float)

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
