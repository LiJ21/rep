from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa


def depth_batches(path: str | Path, levels: int = 5, executable: str | Path | None = None) -> Iterator[pa.RecordBatch]:
    proc = subprocess.Popen(
        [str(_exe(executable)), str(path), "--levels", str(levels), "--format", "ipc"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        yield from pa.ipc.open_stream(proc.stdout)
    finally:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        code = proc.wait()
        if code:
            raise RuntimeError(stderr.strip() or f"orderbook_rs exited with {code}")


def depth_table(path: str | Path, levels: int = 5, executable: str | Path | None = None) -> pa.Table:
    return pa.Table.from_batches(depth_batches(path, levels, executable))


def depth_table_from_arrow(
    rows: pa.Table | Iterable[pa.RecordBatch],
    levels: int = 5,
    executable: str | Path | None = None,
) -> pa.Table:
    data = _ipc_bytes(rows)
    proc = subprocess.run(
        [str(_exe(executable)), "--input-ipc", "--levels", str(levels), "--format", "ipc"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode().strip() or f"orderbook_rs exited with {proc.returncode}")
    return pa.ipc.open_stream(proc.stdout).read_all()


def write_depth_parquet(
    path: str | Path,
    out: str | Path,
    levels: int = 5,
    executable: str | Path | None = None,
) -> None:
    subprocess.run(
        [str(_exe(executable)), str(path), "--levels", str(levels), "--out", str(out)],
        check=True,
    )


def _ipc_bytes(rows: pa.Table | Iterable[pa.RecordBatch]) -> bytes:
    if isinstance(rows, pa.Table):
        batches = rows.to_batches()
        schema = rows.schema
    else:
        batches = list(rows)
        if not batches:
            raise ValueError("rows must contain at least one RecordBatch")
        schema = batches[0].schema

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return sink.getvalue()


def _exe(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path(__file__).resolve().parent / "orderbook_rs" / "target" / "release" / f"orderbook_rs{suffix}"
