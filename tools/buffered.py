from __future__ import annotations

import dataclasses
import hashlib
import os
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from tools.data import Batch, DataSource, Raw, _to_batch
from tools.precision import np_float_dtype

_TRAIN_ROLES = frozenset({"train", "final_train"})
_DONE = object()


@dataclass(frozen=True)
class BufferConfig:
    """Configuration for BufferedDataSource. See prefetch_plan.md.

    workers: producer threads; each streams one date at a time.
    max_buffer_bytes: CreditQueue budget covering queued batches, worker
        in-flight batches, and row-shuffle windows.
    shuffle_dates: permute date order per batches() call (bucketed by size
        when sizes are resolvable, largest bucket first).
    row_shuffle_rows: tumbling-window row shuffle width; 0 disables. Requires
        train_ctx="minimal" because per-row ctx arrays are not permuted.
    seed: base seed for date order and row permutations; None draws one from
        OS entropy at wrapper construction.
    train_ctx: "full" builds the standard per-row ctx; "minimal" emits
        {"n": rows} only (the torch train loop discards ctx).
    charge_dataframe: reserve ~2x per batch so the intermediate polars
        DataFrame is covered while it coexists with the numpy arrays.
    date_sizes: optional callable mapping a date to a size proxy for
        scheduling; when absent, sizes are resolved from the loader's
        path/prod attributes if possible.
    """

    workers: int = 2
    max_buffer_bytes: int = 512 << 20
    shuffle_dates: bool = False
    size_buckets: int = 3
    row_shuffle_rows: int = 0
    seed: int | None = None
    train_ctx: str = "full"
    charge_dataframe: bool = True
    date_sizes: Callable[[str], float] | None = None
    watchdog_interval: float = 10.0
    join_timeout: float = 5.0

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        if self.size_buckets < 1:
            raise ValueError("size_buckets must be positive")
        if self.row_shuffle_rows < 0:
            raise ValueError("row_shuffle_rows must be nonnegative")
        if self.train_ctx not in {"full", "minimal"}:
            raise ValueError("train_ctx must be 'full' or 'minimal'")
        if self.row_shuffle_rows > 0 and self.train_ctx != "minimal":
            raise ValueError(
                "row_shuffle_rows requires train_ctx='minimal': per-row ctx "
                "arrays are not permuted alongside x and y"
            )
        if self.watchdog_interval <= 0:
            raise ValueError("watchdog_interval must be positive")


@dataclass
class _Failure:
    exc: BaseException


class _Stopped(Exception):
    """Internal: the run's stop event fired while a producer was blocked."""


class Reservation:
    """A byte-credit claim against a CreditQueue ledger."""

    __slots__ = ("_queue", "_cost")

    def __init__(self, credit_queue: "CreditQueue", cost: int) -> None:
        self._queue: CreditQueue | None = credit_queue
        self._cost = int(cost)

    @property
    def cost(self) -> int:
        return self._cost

    def resize(self, cost: int) -> None:
        """Correct the claim once the actual size is known.

        Growing never blocks: an overshoot is absorbed and the next reserve()
        call pays for it.
        """
        if self._queue is None:
            raise RuntimeError("cannot resize a released reservation")
        cost = int(cost)
        self._queue._adjust(cost - self._cost)
        self._cost = cost

    def release(self) -> None:
        if self._queue is None:
            return
        credit_queue, self._queue = self._queue, None
        credit_queue._adjust(-self._cost)


class CreditQueue:
    """MPSC queue whose memory is bounded by a byte-credit ledger.

    Structure (an unbounded deque) is separate from accounting: producers
    reserve() credits before materializing an item, the credit travels with
    the enqueued item, and the consumer releases it once the item is no
    longer held. Sentinels enqueued without a reservation bypass the ledger,
    so shutdown signaling can never deadlock against a full budget. A
    reservation larger than the whole budget is admitted only when the ledger
    is empty.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._cond = threading.Condition()
        self._items: deque[tuple[Any, Reservation | None]] = deque()
        self._used = 0

    @property
    def used_bytes(self) -> int:
        with self._cond:
            return self._used

    def reserve(
        self,
        cost: int,
        stop: threading.Event | None = None,
        poll: float = 0.5,
    ) -> Reservation | None:
        """Claim cost bytes, blocking while the budget is exhausted.

        Returns None if stop is set while waiting.
        """
        cost = int(cost)
        with self._cond:
            while self._used > 0 and self._used + cost > self.max_bytes:
                if stop is not None and stop.is_set():
                    return None
                self._cond.wait(timeout=poll if stop is not None else None)
            self._used += cost
            return Reservation(self, cost)

    def _adjust(self, delta: int) -> None:
        with self._cond:
            self._used += delta
            if delta < 0:
                self._cond.notify_all()

    def put(self, item: Any, reservation: Reservation | None = None) -> None:
        with self._cond:
            self._items.append((item, reservation))
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> tuple[Any, Reservation | None]:
        with self._cond:
            while not self._items:
                if not self._cond.wait(timeout=timeout):
                    raise queue.Empty
            return self._items.popleft()

    def drain(self) -> None:
        """Discard all queued items and release their credits (teardown)."""
        with self._cond:
            items = list(self._items)
            self._items.clear()
        for _, reservation in items:
            if reservation is not None:
                reservation.release()


class _DirectEmitter:
    def __init__(self, run: "_Run") -> None:
        self._run = run

    def push(self, batch: Batch, reservation: Reservation) -> None:
        self._run.queue.put(batch, reservation)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _RowShuffleEmitter:
    """Tumbling-window row shuffle over converted numpy batches.

    Incoming rows are copied into a preallocated window; when it fills, a
    seeded permutation re-slices it into output batches. Rows never cross
    dates (one emitter per date task). The window holds one long-lived
    reservation; each emitted batch carries its own.
    """

    def __init__(self, run: "_Run", rows: int, rng: np.random.Generator) -> None:
        self._run = run
        self._rows = rows
        self._rng = rng
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._filled = 0
        self._reservation: Reservation | None = None

    def push(self, batch: Batch, reservation: Reservation) -> None:
        x, y, _ = batch
        try:
            if self._x is None:
                row_bytes = x.itemsize * x.shape[1] + y.dtype.itemsize
                self._reservation = self._run.reserve(self._rows * row_bytes)
                self._x = np.empty((self._rows, x.shape[1]), dtype=x.dtype)
                self._y = np.empty(self._rows, dtype=y.dtype)
            offset = 0
            while offset < len(y):
                take = min(self._rows - self._filled, len(y) - offset)
                stop = self._filled + take
                self._x[self._filled : stop] = x[offset : offset + take]
                self._y[self._filled : stop] = y[offset : offset + take]
                self._filled = stop
                offset += take
                if self._filled == self._rows:
                    self._emit()
        finally:
            reservation.release()

    def _emit(self) -> None:
        assert self._x is not None and self._y is not None
        perm = self._rng.permutation(self._filled)
        out_rows = self._run.batch_size or self._filled
        row_bytes = self._x.itemsize * self._x.shape[1] + self._y.dtype.itemsize
        for start in range(0, self._filled, out_rows):
            idx = perm[start : start + out_rows]
            reservation = self._run.reserve(int(idx.size) * row_bytes)
            batch = (self._x[idx], self._y[idx], {"n": int(idx.size)})
            self._run.queue.put(batch, reservation)
        self._filled = 0

    def flush(self) -> None:
        if self._filled:
            self._emit()

    def close(self) -> None:
        if self._reservation is not None:
            self._reservation.release()
        self._x = self._y = None


class _Run:
    """One batches() invocation: producer pool, credit queue, lifecycle.

    Producers pull dates from a pre-filled queue, stream each date through a
    throwaway single-date DataSource, and push converted batches through the
    CreditQueue. Exactly one of DONE / _Failure is enqueued per claimed date;
    the consumer finishes at len(dates) DONEs, re-raises on _Failure, and a
    watchdog turns dead-producer protocol bugs into errors instead of hangs.
    """

    def __init__(
        self,
        source: DataSource,
        config: BufferConfig,
        dates: list[str],
        batch_size: int | None,
        multicollect: int,
        workers: int,
        base_seed: int,
        run_index: int,
    ) -> None:
        self.source = source
        self.config = config
        self.dates = dates
        self.batch_size = batch_size
        self.multicollect = multicollect
        self.workers = workers
        self.base_seed = base_seed
        self.run_index = run_index
        self.queue = CreditQueue(config.max_buffer_bytes)
        self.stop = threading.Event()
        self.date_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        for date in dates:
            self.date_queue.put(date)
        self.threads: list[threading.Thread] = []
        self.minimal_ctx = config.train_ctx == "minimal"

    def reserve(self, cost: int) -> Reservation:
        reservation = self.queue.reserve(cost, stop=self.stop)
        if reservation is None:
            raise _Stopped()
        return reservation

    def _start(self) -> None:
        for i in range(self.workers):
            thread = threading.Thread(
                target=self._worker,
                name=f"buffered-w{i}-{id(self):x}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                date = self.date_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._produce_date(date)
            except _Stopped:
                return
            except BaseException as exc:  # noqa: BLE001 - forwarded to consumer
                self.queue.put(_Failure(exc))
                return
            self.queue.put(_DONE)

    def _produce_date(self, date: str) -> None:
        sub = dataclasses.replace(
            self.source, dates=[date], cache=None, cache_key=None
        )
        emitter = self._emitter(date)
        gen = sub.dataframe_batches(self.batch_size, self.multicollect)
        try:
            while True:
                reservation = self.reserve(self._estimated_cost())
                try:
                    df = next(gen)
                except StopIteration:
                    reservation.release()
                    break
                except BaseException:
                    reservation.release()
                    raise
                batch = self._convert(df)
                del df
                reservation.resize(_batch_nbytes(batch))
                emitter.push(batch, reservation)
            emitter.flush()
        finally:
            emitter.close()
            gen.close()

    def _emitter(self, date: str) -> Any:
        if self.config.row_shuffle_rows <= 0:
            return _DirectEmitter(self)
        rng = np.random.default_rng(
            _stable_seed(self.base_seed, self.run_index, date)
        )
        return _RowShuffleEmitter(self, self.config.row_shuffle_rows, rng)

    def _convert(self, df: pl.DataFrame) -> Batch:
        if self.minimal_ctx:
            x = df.select(self.source.features).to_numpy()
            y = df.get_column(self.source.target).to_numpy()
            return x, y, {"n": df.height}
        return _to_batch(df, self.source.features, self.source.target)

    def _estimated_cost(self) -> int:
        rows = self.batch_size or 0
        if rows <= 0:
            return 0
        width = len(self.source.features) + 1
        itemsize = np.dtype(np_float_dtype(self.source.precision)).itemsize
        row_bytes = width * itemsize + (0 if self.minimal_ctx else 16)
        return rows * row_bytes * (2 if self.config.charge_dataframe else 1)

    def iterate(self) -> Iterator[Batch]:
        from tqdm import tqdm

        total = len(self.dates)
        done = 0
        held: Reservation | None = None
        self._start()
        try:
            with tqdm(desc="Loading data", unit="row", unit_scale=True) as bar:
                while done < total:
                    try:
                        item, reservation = self.queue.get(
                            timeout=self.config.watchdog_interval
                        )
                    except queue.Empty:
                        if any(t.is_alive() for t in self.threads):
                            continue
                        try:
                            item, reservation = self.queue.get(timeout=0)
                        except queue.Empty:
                            raise RuntimeError(
                                "buffered producers exited with "
                                f"{total - done} of {total} dates unfinished"
                            ) from None
                    if item is _DONE:
                        done += 1
                        continue
                    if isinstance(item, _Failure):
                        raise item.exc
                    if held is not None:
                        held.release()
                    held = reservation
                    bar.update(item[2].get("n", len(item[1])))
                    yield item
        finally:
            self.stop.set()
            if held is not None:
                held.release()
            self.queue.drain()
            for thread in self.threads:
                thread.join(timeout=self.config.join_timeout)


class BufferedDataSource:
    """Prefetching, optionally shuffling wrapper around a DataSource.

    Overrides batches() with a multi-producer prefetch pipeline (one date per
    producer task; within-date order preserved unless row shuffle is on) and
    delegates everything else to the wrapped source. Memory is bounded by
    config.max_buffer_bytes via a byte-credit ledger. See prefetch_plan.md.
    """

    def __init__(self, source: DataSource, config: BufferConfig) -> None:
        self._source = source
        self._config = config
        seed = config.seed
        if seed is None:
            seed = int(np.random.SeedSequence().entropy) % (2**63)
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(self._base_seed)
        self._run_lock = threading.Lock()
        self._run_index = 0

    def __getattr__(self, name: str) -> Any:
        try:
            source = self.__dict__["_source"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(source, name)

    def __repr__(self) -> str:
        return f"BufferedDataSource({self._source!r}, {self._config!r})"

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def config(self) -> BufferConfig:
        return self._config

    @property
    def is_shuffled(self) -> bool:
        return self._config.shuffle_dates or self._config.row_shuffle_rows > 0

    @property
    def is_deterministic(self) -> bool:
        """True when repeated batches() calls yield the identical stream."""
        return self._config.workers <= 1 and not self.is_shuffled

    def with_transform(self, transform: Any) -> "BufferedDataSource":
        return BufferedDataSource(
            self._source.with_transform(transform), self._config
        )

    def batches(
        self, batch_size: int | None = None, multicollect: int = -1
    ) -> Iterator[Batch]:
        with self._run_lock:
            run_index = self._run_index
            self._run_index += 1
            order_seed = int(self._rng.integers(2**63))
        dates = self._date_order(order_seed)
        run = _Run(
            source=self._source,
            config=self._config,
            dates=dates,
            batch_size=batch_size,
            multicollect=multicollect,
            workers=self._effective_workers(len(dates)),
            base_seed=self._base_seed,
            run_index=run_index,
        )
        return run.iterate()

    def _effective_workers(self, n_dates: int) -> int:
        workers = min(self._config.workers, max(1, n_dates))
        if self._source.polars_engine == "gpu":
            workers = 1
        if _loader_attr(self._source.loader, "l2_depth") is not None:
            workers = 1
        return workers

    def _date_order(self, seed: int) -> list[str]:
        dates = list(self._source.dates)
        if not self._config.shuffle_dates or len(dates) < 2:
            return dates
        rng = np.random.default_rng(seed)
        sizes = self._date_size_map(dates)
        if sizes is None:
            return [dates[i] for i in rng.permutation(len(dates))]
        ordered = sorted(dates, key=lambda d: -sizes[d])
        out: list[str] = []
        n_buckets = min(self._config.size_buckets, len(ordered))
        for bucket in np.array_split(np.arange(len(ordered)), n_buckets):
            for i in bucket[rng.permutation(len(bucket))]:
                out.append(ordered[i])
        return out

    def _date_size_map(self, dates: list[str]) -> dict[str, float] | None:
        size_fn = self._config.date_sizes
        if size_fn is None:
            path = _loader_attr(self._source.loader, "path")
            prod = _loader_attr(self._source.loader, "prod")
            if path is None or prod is None:
                return None

            def size_fn(date: str) -> float:
                return float(os.path.getsize(Raw.resolve_path(date, prod, path)[0]))

        try:
            return {date: float(size_fn(date)) for date in dates}
        except Exception:
            return None


def buffered_wrapper(
    train: BufferConfig | None,
    other: BufferConfig | None = BufferConfig(workers=1),
) -> Callable[[DataSource, str], Any]:
    """Build a Pipeline.data_source_wrapper applying configs by role.

    train/final_train get the train config; every other role (val, final_val,
    test, fit) gets other, which defaults to order-preserving single-worker
    prefetch. Pass None for either to leave those roles unwrapped. The hook
    cannot see the adapter: XGBoost streaming additionally enforces an
    unshuffled, deterministic source at the adapter (tools/model.py).
    """

    def wrap(source: DataSource, role: str) -> Any:
        config = train if role in _TRAIN_ROLES else other
        if config is None:
            return source
        return BufferedDataSource(source, config)

    return wrap


def _batch_nbytes(batch: Batch) -> int:
    x, y, ctx = batch
    total = x.nbytes + y.nbytes
    for value in ctx.values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def _stable_seed(base_seed: int, run_index: int, date: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}|{run_index}|{date}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _loader_attr(loader: Any, name: str) -> Any:
    seen: set[int] = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        value = getattr(loader, name, None)
        if value is not None:
            return value
        loader = getattr(loader, "loader", None)
    return None
