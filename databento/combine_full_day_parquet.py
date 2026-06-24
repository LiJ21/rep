#!/usr/bin/env python3
"""Build full-day parquet files from Databento MBO gap DBNs and existing parquet windows."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl


DEFAULT_PARTIAL_DIR = Path("data/databento_glbx_mdp3_mbo")
DEFAULT_GAP_DIR = Path("data/databento_glbx_mdp3_mbo_full_utc_day")
DEFAULT_OUT_DIR = Path("data/databento_glbx_mdp3_mbo_full_day_parquet")

PARTIAL_RE = re.compile(
    r"^(?P<symbol>[^_]+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<tag>.+)\.parquet$"
)
GAP_LABELS = ("gap_before", "gap_after")


@dataclass(frozen=True)
class PartialFile:
    path: Path
    symbol: str
    day: str
    tag: str

    @property
    def base(self) -> str:
        return f"{self.symbol}_{self.day}_{self.tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate gap-before, existing-window, and gap-after parquet into full-day parquet files"
    )
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--gap-dir", type=Path, default=DEFAULT_GAP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", help="First date to include, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Last date to include, YYYY-MM-DD")
    parser.add_argument("--symbol", action="append", help="Symbol to include, e.g. ESM6. Can be repeated")
    parser.add_argument("--price-type", choices=["fixed", "float"], default="fixed")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--sort-by", choices=["none", "ts_recv", "ts_event"], default="none")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full-day output files")
    parser.add_argument("--no-convert-gaps", action="store_true", help="Require gap parquet files to already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without converting or writing")
    parser.add_argument("--limit", type=int, help="Process at most this many partial files, useful for smoke tests")
    return parser.parse_args()


def parse_partial_file(path: Path) -> PartialFile | None:
    match = PARTIAL_RE.match(path.name)
    if not match:
        return None
    return PartialFile(
        path=path,
        symbol=match.group("symbol"),
        day=match.group("day"),
        tag=match.group("tag"),
    )


def find_partials(args: argparse.Namespace) -> list[PartialFile]:
    partials: list[PartialFile] = []
    symbols = set(args.symbol or [])
    for path in sorted(args.partial_dir.glob("*.parquet")):
        partial = parse_partial_file(path)
        if partial is None:
            continue
        if args.start_date and partial.day < args.start_date:
            continue
        if args.end_date and partial.day > args.end_date:
            continue
        if symbols and partial.symbol not in symbols:
            continue
        partials.append(partial)
    if args.limit is not None:
        partials = partials[: args.limit]
    return partials


def gap_dbn_path(gap_dir: Path, partial: PartialFile, label: str) -> Path:
    return gap_dir / f"{partial.base}_{label}.dbn.zst"


def gap_parquet_path(gap_dir: Path, partial: PartialFile, label: str) -> Path:
    return gap_dir / f"{partial.base}_{label}.parquet"


def output_path(out_dir: Path, partial: PartialFile) -> Path:
    return out_dir / f"{partial.base}_full_day.parquet"


def convert_gap_dbn(src: Path, dest: Path, price_type: str) -> None:
    import databento as db

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        store = db.DBNStore.from_file(src)
        store.to_parquet(tmp, price_type=price_type, mode="x")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def ensure_gap_parquet(args: argparse.Namespace, partial: PartialFile, label: str) -> Path:
    parquet_path = gap_parquet_path(args.gap_dir, partial, label)
    if parquet_path.exists():
        return parquet_path

    if args.no_convert_gaps:
        raise FileNotFoundError(f"Missing required gap parquet: {parquet_path}")

    dbn_path = gap_dbn_path(args.gap_dir, partial, label)
    if not dbn_path.exists():
        raise FileNotFoundError(f"Missing required gap DBN: {dbn_path}")

    if args.dry_run:
        print(f"would convert {dbn_path} -> {parquet_path}")
        return parquet_path

    print(f"converting {dbn_path.name} -> {parquet_path.name}")
    convert_gap_dbn(dbn_path, parquet_path, args.price_type)
    return parquet_path


def write_full_day(source_paths: list[Path], dest: Path, sort_by: str, compression: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    scans = [pl.scan_parquet(path) for path in source_paths]
    frame = pl.concat(scans, how="vertical")
    if sort_by != "none":
        frame = frame.sort(sort_by)
    try:
        frame.sink_parquet(tmp, compression=compression)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    args = parse_args()
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")

    partials = find_partials(args)
    if not partials:
        raise SystemExit("No partial parquet files matched the selected filters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"matched {len(partials)} partial parquet file(s)")

    written = 0
    skipped = 0
    for partial in partials:
        dest = output_path(args.out_dir, partial)
        if dest.exists() and not args.overwrite:
            print(f"skipped {dest} (exists)")
            skipped += 1
            continue
        if dest.exists() and args.overwrite and not args.dry_run:
            dest.unlink()

        before = ensure_gap_parquet(args, partial, "gap_before")
        after = ensure_gap_parquet(args, partial, "gap_after")
        source_paths = [before, partial.path, after]

        if args.dry_run:
            joined = ", ".join(str(path) for path in source_paths)
            print(f"would write {dest} from {joined}")
            written += 1
            continue

        print(f"writing {dest}")
        write_full_day(source_paths, dest, args.sort_by, args.compression)
        written += 1

    print(f"done: {written} planned/written, {skipped} skipped")


if __name__ == "__main__":
    main()
