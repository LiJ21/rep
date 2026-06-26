from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    def start_run(self, config: dict[str, Any]) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@dataclass
class NullTracker:
    def start_run(self, config: dict[str, Any]) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


@dataclass
class WandbTracker:
    project: str
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    _run: Any = field(default=None, init=False, repr=False)

    def start_run(self, config: dict[str, Any]) -> None:
        import wandb

        merged = {**self.config, **config}
        self._run = wandb.init(project=self.project, name=self.name, config=merged, **self.kwargs)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(params, allow_val_change=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
