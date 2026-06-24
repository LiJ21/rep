#!/usr/bin/env python3
"""Extract metadata and verify Databento full-day parquet integrity."""

import argparse
import json
import re
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import databento as db
import polars as pl

from databento_auth import DEFAULT_API_KEY_FILE, read_databento_api_key


DEFAULT_PARTIAL_DIR = Path("data/databento_glbx_mdp3_mbo")
DEFAULT_GAP_DIR = Path("data/databento_glbx_mdp3_mbo_full_utc_day")
DEFAULT_FULL_DIR = Path("data/databento_glbx_mdp3_mbo_full_day_parquet")

DATASET = "GLBX.MDP3"
SCHEMA = "mbo"
STYPE_IN = "raw_symbol"

EXISTING_START = time(13, 15)
EXISTING_END = time(20, 5)

F_MAYBE_BAD_BOOK = 4
F_BAD_TS_RECV = 8
F_SNAPSHOT = 32
F_LAST = 128

FULL_RE = re.compile(
    r"^(?P<symbol>[^_]+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<tag>.+)_full_day\.parquet$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DBN/parquet metadata and verify local full-day Databento MBO parquet files"
        )
    )
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--gap-dir", type=Path, default=DEFAULT_GAP_DIR)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL_DIR)
    parser.add_argument("--start-date", help="First date to include, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Last date to include, YYYY-MM-DD")
    parser.add_argument("--symbol", action="append", help="Symbol to include, e.g. ESM6. Can be repeated")
    parser.add_argument("--limit", type=int, help="Process at most this many full-day files")
    parser.add_argument("--report", type=Path, help="Write a JSON metadata/integrity report")
    parser.add_argument("--zstd-test", action="store_true", help="Run `zstd -t` on source .dbn.zst files")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use Databento metadata API to fetch dataset condition and expected record counts",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help="Encrypted GPG file containing the Databento API key for --online",
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--schema", default=SCHEMA)
    parser.add_argument("--stype-in", default=STYPE_IN)
    return parser.parse_args()


def utc_dt(day: str, value: time = time(0, 0)) -> datetime:
    return datetime.combine(date.fromisoformat(day), value, tzinfo=timezone.utc)


def next_utc_day(day: str) -> datetime:
    return utc_dt(day) + timedelta(days=1)


def dt_to_ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def find_full_files(args: argparse.Namespace) -> list[Path]:
    symbols = set(args.symbol or [])
    files: list[Path] = []
    for path in sorted(args.full_dir.glob("*_full_day.parquet")):
        match = FULL_RE.match(path.name)
        if not match:
            continue
        day = match.group("day")
        symbol = match.group("symbol")
        if args.start_date and day < args.start_date:
            continue
        if args.end_date and day > args.end_date:
            continue
        if symbols and symbol not in symbols:
            continue
        files.append(path)
    if args.limit is not None:
        files = files[: args.limit]
    return files


def collect_schema(path: Path) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in pl.scan_parquet(path).collect_schema().items()}


def parquet_bounds(path: Path) -> dict[str, Any]:
    schema = collect_schema(path)
    lf = pl.scan_parquet(path)
    exprs: list[pl.Expr] = [pl.len().alias("rows")]

    if "ts_recv" in schema:
        exprs.extend(
            [
                pl.col("ts_recv").first().alias("first_recv"),
                pl.col("ts_recv").last().alias("last_recv"),
                pl.col("ts_recv").min().alias("min_recv"),
                pl.col("ts_recv").max().alias("max_recv"),
            ]
        )
    if "ts_event" in schema:
        exprs.extend(
            [
                pl.col("ts_event").min().alias("min_event"),
                pl.col("ts_event").max().alias("max_event"),
            ]
        )

    result = lf.select(*exprs).collect().to_dicts()[0]
    return {"schema": schema, **{key: iso(value) for key, value in result.items()}}


def full_parquet_stats(path: Path, start: datetime) -> dict[str, Any]:
    stats = parquet_bounds(path)
    schema = stats["schema"]
    lf = pl.scan_parquet(path)
    exprs: list[pl.Expr] = []

    if "ts_recv" in schema:
        recv_ns = pl.col("ts_recv").cast(pl.Int64)
        exprs.extend(
            [
                (recv_ns.diff() < 0).sum().alias("ts_recv_regressions"),
                (pl.col("ts_recv") == start).sum().alias("snapshot_rows"),
            ]
        )
    if "ts_event" in schema:
        event_ns = pl.col("ts_event").cast(pl.Int64)
        exprs.append((event_ns.diff() < 0).sum().alias("ts_event_regressions"))
    if "sequence" in schema:
        seq = pl.col("sequence").cast(pl.Int64)
        exprs.extend(
            [
                seq.min().alias("sequence_min"),
                seq.max().alias("sequence_max"),
                seq.n_unique().alias("sequence_unique"),
            ]
        )
    if "flags" in schema:
        flags = pl.col("flags").cast(pl.UInt16)
        snapshot_rows = pl.col("ts_recv") == start if "ts_recv" in schema else pl.lit(False)
        exprs.extend(
            [
                ((flags & F_MAYBE_BAD_BOOK) != 0).sum().alias("maybe_bad_book_rows"),
                (((flags & F_BAD_TS_RECV) != 0) & ((flags & F_SNAPSHOT) == 0))
                .sum()
                .alias("non_snapshot_bad_ts_recv_rows"),
                (
                    snapshot_rows
                    & (((flags & F_SNAPSHOT) == 0) | ((flags & F_BAD_TS_RECV) == 0))
                )
                .sum()
                .alias("snapshot_flag_mismatch_rows"),
                (snapshot_rows & ((flags & F_LAST) != 0)).sum().alias("snapshot_last_rows"),
            ]
        )
    if "action" in schema and "ts_recv" in schema:
        exprs.append(((pl.col("ts_recv") == start) & (pl.col("action") == "R")).sum().alias("snapshot_reset_rows"))

    if exprs:
        stats.update({key: iso(value) for key, value in lf.select(*exprs).collect().to_dicts()[0].items()})

    if "sequence" in schema and "ts_recv" in schema:
        seq = pl.col("sequence").cast(pl.Int64)
        post = (
            lf.filter(pl.col("ts_recv") > start)
            .select(
                (seq.diff() < 0).sum().alias("post_snapshot_sequence_regressions"),
                (seq.diff() > 1).sum().alias("post_snapshot_sequence_positive_jumps"),
            )
            .collect()
            .to_dicts()[0]
        )
        stats.update(post)

    return stats


def dbn_metadata(path: Path) -> dict[str, Any]:
    info = file_info(path)
    if not path.exists():
        return info

    store = db.DBNStore.from_file(path)
    metadata = store.metadata
    info["metadata"] = {
        "version": metadata.version,
        "dataset": metadata.dataset,
        "schema": str(metadata.schema),
        "start_ns": metadata.start,
        "end_ns": metadata.end,
        "limit": metadata.limit,
        "stype_in": str(metadata.stype_in),
        "stype_out": str(metadata.stype_out),
        "ts_out": metadata.ts_out,
        "symbols": list(metadata.symbols),
        "partial": list(metadata.partial),
        "not_found": list(metadata.not_found),
        "mappings": metadata.mappings,
    }
    if metadata.start is not None:
        info["metadata"]["start"] = datetime.fromtimestamp(
            metadata.start / 1_000_000_000, tz=timezone.utc
        ).isoformat()
    if metadata.end is not None:
        info["metadata"]["end"] = datetime.fromtimestamp(
            metadata.end / 1_000_000_000, tz=timezone.utc
        ).isoformat()
    return info


def zstd_test(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"checked": False, "ok": False, "error": "missing"}
    result = subprocess.run(
        ["zstd", "-q", "-t", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "checked": True,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def add_issue(issues: list[dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def check_segment_bounds(
    issues: list[dict[str, str]],
    label: str,
    stats: dict[str, Any],
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    rows = stats.get("rows", 0)
    if rows == 0:
        add_issue(issues, "warning", f"{label} has zero rows")
        return

    min_recv = stats.get("min_recv")
    max_recv = stats.get("max_recv")
    if min_recv is None or max_recv is None:
        add_issue(issues, "error", f"{label} is missing ts_recv bounds")
        return

    min_recv_dt = datetime.fromisoformat(min_recv)
    max_recv_dt = datetime.fromisoformat(max_recv)
    if min_recv_dt < expected_start:
        add_issue(issues, "error", f"{label} starts before expected range: {min_recv}")
    if max_recv_dt >= expected_end:
        add_issue(issues, "error", f"{label} ends at/after exclusive range end: {max_recv}")


def expected_ranges(day: str) -> dict[str, tuple[datetime, datetime]]:
    start = utc_dt(day)
    existing_start = utc_dt(day, EXISTING_START)
    existing_end = utc_dt(day, EXISTING_END)
    end = next_utc_day(day)
    return {
        "gap_before": (start, existing_start),
        "partial": (existing_start, existing_end),
        "gap_after": (existing_end, end),
        "full": (start, end),
    }


def base_paths(args: argparse.Namespace, base: str) -> dict[str, Path]:
    return {
        "gap_before": args.gap_dir / f"{base}_gap_before.parquet",
        "partial": args.partial_dir / f"{base}.parquet",
        "gap_after": args.gap_dir / f"{base}_gap_after.parquet",
    }


def dbn_paths(args: argparse.Namespace, base: str) -> dict[str, Path]:
    return {
        "gap_before": args.gap_dir / f"{base}_gap_before.dbn.zst",
        "gap_after": args.gap_dir / f"{base}_gap_after.dbn.zst",
    }


def verify_one(
    args: argparse.Namespace,
    full_path: Path,
    online_counts: dict[str, int] | None,
    online_conditions: dict[str, str | None] | None,
) -> dict[str, Any]:
    match = FULL_RE.match(full_path.name)
    if not match:
        raise ValueError(f"Unexpected full-day parquet filename: {full_path}")

    symbol = match.group("symbol")
    day = match.group("day")
    tag = match.group("tag")
    base = f"{symbol}_{day}_{tag}"
    ranges = expected_ranges(day)
    issues: list[dict[str, str]] = []

    item: dict[str, Any] = {
        "base": base,
        "symbol": symbol,
        "day": day,
        "tag": tag,
        "full": {"file": file_info(full_path)},
        "segments": {},
        "dbn": {},
        "issues": issues,
    }

    if not full_path.exists():
        add_issue(issues, "error", f"missing full parquet: {full_path}")
        return item

    segment_paths = base_paths(args, base)
    missing = [label for label, path in segment_paths.items() if not path.exists()]
    for label in missing:
        add_issue(issues, "error", f"missing {label} parquet: {segment_paths[label]}")
    if missing:
        return item

    full_stats = full_parquet_stats(full_path, ranges["full"][0])
    item["full"]["parquet"] = full_stats

    previous_label = None
    previous_max: datetime | None = None
    segment_rows = 0
    segment_schema: dict[str, str] | None = None
    for label, path in segment_paths.items():
        stats = parquet_bounds(path)
        item["segments"][label] = {"file": file_info(path), "parquet": stats}
        segment_rows += stats.get("rows", 0)
        check_segment_bounds(issues, label, stats, *ranges[label])

        if segment_schema is None:
            segment_schema = stats["schema"]
        elif stats["schema"] != segment_schema:
            add_issue(issues, "error", f"{label} schema differs from previous segments")

        if full_stats.get("schema") != stats["schema"]:
            add_issue(issues, "error", f"{label} schema differs from full output")

        min_recv = stats.get("min_recv")
        max_recv = stats.get("max_recv")
        if previous_max is not None and min_recv is not None and previous_label is not None:
            min_recv_dt = datetime.fromisoformat(min_recv)
            if min_recv_dt < previous_max:
                add_issue(
                    issues,
                    "error",
                    f"{previous_label} overlaps {label}: {previous_max.isoformat()} > {min_recv}",
                )
        previous_label = label
        previous_max = datetime.fromisoformat(max_recv) if max_recv is not None else previous_max

    if full_stats.get("rows") != segment_rows:
        add_issue(
            issues,
            "error",
            f"full row count {full_stats.get('rows')} != segment row sum {segment_rows}",
        )
    item["segment_row_sum"] = segment_rows

    full_start, full_end = ranges["full"]
    min_recv = full_stats.get("min_recv")
    max_recv = full_stats.get("max_recv")
    if min_recv is None or max_recv is None:
        add_issue(issues, "error", "full parquet is missing ts_recv bounds")
    else:
        min_recv_dt = datetime.fromisoformat(min_recv)
        max_recv_dt = datetime.fromisoformat(max_recv)
        if min_recv_dt != full_start:
            add_issue(issues, "error", f"full parquet does not start at UTC midnight: {min_recv}")
        if max_recv_dt >= full_end:
            add_issue(issues, "error", f"full parquet extends past UTC day: {max_recv}")

    if full_stats.get("ts_recv_regressions", 0):
        add_issue(issues, "error", f"ts_recv regressions: {full_stats['ts_recv_regressions']}")
    if full_stats.get("post_snapshot_sequence_regressions", 0):
        add_issue(
            issues,
            "error",
            f"post-snapshot sequence regressions: {full_stats['post_snapshot_sequence_regressions']}",
        )
    if full_stats.get("maybe_bad_book_rows", 0):
        add_issue(issues, "error", f"F_MAYBE_BAD_BOOK rows: {full_stats['maybe_bad_book_rows']}")
    if full_stats.get("non_snapshot_bad_ts_recv_rows", 0):
        add_issue(
            issues,
            "warning",
            f"non-snapshot F_BAD_TS_RECV rows: {full_stats['non_snapshot_bad_ts_recv_rows']}",
        )
    if full_stats.get("snapshot_rows", 0) == 0:
        add_issue(issues, "error", "missing midnight snapshot rows")
    if full_stats.get("snapshot_flag_mismatch_rows", 0):
        add_issue(
            issues,
            "error",
            f"snapshot rows missing F_SNAPSHOT/F_BAD_TS_RECV: {full_stats['snapshot_flag_mismatch_rows']}",
        )
    if full_stats.get("snapshot_reset_rows", 0) != 1:
        add_issue(issues, "warning", f"snapshot reset rows: {full_stats.get('snapshot_reset_rows')}")
    if full_stats.get("snapshot_last_rows", 0) != 1:
        add_issue(issues, "warning", f"snapshot F_LAST rows: {full_stats.get('snapshot_last_rows')}")

    for label, path in dbn_paths(args, base).items():
        try:
            metadata = dbn_metadata(path)
            item["dbn"][label] = metadata
            if not metadata.get("exists"):
                add_issue(issues, "error", f"missing {label} DBN: {path}")
                continue
            md = metadata.get("metadata", {})
            expected_start, expected_end = ranges[label]
            if md.get("dataset") != args.dataset:
                add_issue(issues, "error", f"{label} DBN dataset mismatch: {md.get('dataset')}")
            if str(md.get("schema")).lower() != args.schema.lower():
                add_issue(issues, "error", f"{label} DBN schema mismatch: {md.get('schema')}")
            if symbol not in md.get("symbols", []):
                add_issue(issues, "error", f"{label} DBN symbols do not contain {symbol}: {md.get('symbols')}")
            if md.get("start_ns") != dt_to_ns(expected_start):
                add_issue(issues, "error", f"{label} DBN start mismatch: {md.get('start')}")
            if md.get("end_ns") != dt_to_ns(expected_end):
                add_issue(issues, "error", f"{label} DBN end mismatch: {md.get('end')}")
            if args.zstd_test:
                item["dbn"][label]["zstd_test"] = zstd_test(path)
                if not item["dbn"][label]["zstd_test"]["ok"]:
                    add_issue(issues, "error", f"{label} zstd test failed")
        except Exception as exc:
            add_issue(issues, "error", f"failed to read {label} DBN metadata: {exc}")

    if online_counts is not None:
        key = f"{symbol}|{day}"
        expected = online_counts.get(key)
        item["online_record_count"] = expected
        if expected is not None and expected != full_stats.get("rows"):
            add_issue(
                issues,
                "warning",
                f"Databento record count {expected} != local full rows {full_stats.get('rows')}",
            )

    if online_conditions is not None:
        condition = online_conditions.get(day)
        item["online_dataset_condition"] = condition
        if condition != "available":
            add_issue(issues, "warning", f"Databento dataset condition for {day}: {condition}")

    return item


def online_metadata(
    args: argparse.Namespace,
    selected: list[Path],
) -> tuple[dict[str, str | None], dict[str, int]]:
    client = db.Historical(read_databento_api_key(args.api_key_file))
    days = sorted({FULL_RE.match(path.name).group("day") for path in selected if FULL_RE.match(path.name)})
    if not days:
        return {}, {}

    conditions_raw = client.metadata.get_dataset_condition(
        dataset=args.dataset,
        start_date=days[0],
        end_date=days[-1],
    )
    conditions = {row["date"]: row.get("condition") for row in conditions_raw}

    counts: dict[str, int] = {}
    for path in selected:
        match = FULL_RE.match(path.name)
        if not match:
            continue
        symbol = match.group("symbol")
        day = match.group("day")
        end_day = date.fromisoformat(day) + timedelta(days=1)
        counts[f"{symbol}|{day}"] = client.metadata.get_record_count(
            dataset=args.dataset,
            symbols=symbol,
            schema=args.schema,
            stype_in=args.stype_in,
            start=f"{day}T00:00:00Z",
            end=f"{end_day.isoformat()}T00:00:00Z",
        )
    return conditions, counts


def main() -> None:
    args = parse_args()
    if args.start_date:
        date.fromisoformat(args.start_date)
    if args.end_date:
        date.fromisoformat(args.end_date)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")

    full_files = find_full_files(args)
    if not full_files:
        raise SystemExit("No full-day parquet files matched the selected filters")

    dataset_conditions: dict[str, str | None] = {}
    online_counts: dict[str, int] | None = None
    if args.online:
        dataset_conditions, online_counts = online_metadata(args, full_files)

    conditions_for_files = dataset_conditions if args.online else None
    results = [verify_one(args, path, online_counts, conditions_for_files) for path in full_files]
    errors = sum(1 for item in results for issue in item["issues"] if issue["severity"] == "error")
    warnings = sum(1 for item in results for issue in item["issues"] if issue["severity"] == "warning")

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "summary": {
            "files_checked": len(results),
            "errors": errors,
            "warnings": warnings,
            "online": args.online,
            "zstd_test": args.zstd_test,
        },
        "dataset_conditions": dataset_conditions,
        "files": results,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=json_default) + "\n")

    print(f"checked {len(results)} full-day parquet file(s)")
    print(f"errors: {errors}")
    print(f"warnings: {warnings}")
    if args.report:
        print(f"wrote report: {args.report}")

    shown = 0
    for item in results:
        if not item["issues"]:
            continue
        print(item["base"])
        for issue in item["issues"]:
            print(f"  {issue['severity']}: {issue['message']}")
            shown += 1
            if shown >= 40:
                remaining = errors + warnings - shown
                if remaining > 0:
                    print(f"  ... {remaining} more issue(s), see report")
                raise SystemExit(1 if errors else 0)

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
