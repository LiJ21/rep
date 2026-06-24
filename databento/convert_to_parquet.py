#!/usr/bin/env python3
"""Convert Databento DBN/zst files to Parquet."""

import argparse
from pathlib import Path

import databento as db


def convert_file(src: Path, out_dir: Path, price_type: str, overwrite: bool) -> Path:
    dest = out_dir / src.name.replace(".dbn.zst", ".parquet")
    store = db.DBNStore.from_file(src)
    # Streams to Parquet in 64K-row chunks via pyarrow's ParquetWriter — never
    # materializes the whole file as a single pandas DataFrame (these MBO files
    # decompress to hundreds of millions of rows).
    store.to_parquet(
        dest,
        price_type=price_type,
        mode="w" if overwrite else "x",
    )
    return dest


def main():
    parser = argparse.ArgumentParser(description="Convert .dbn.zst files to Parquet")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/databento_glbx_mdp3_mbo_full_utc_day",
        help="Path to a .dbn.zst file or directory of them",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to same directory as input files)",
    )
    parser.add_argument(
        "--price-type",
        default="fixed",
        choices=["fixed", "float", "decimal"],
        help="Price representation: 'fixed' (int64 1e-9, lossless/compact), "
        "'float', or 'decimal' (default: fixed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .parquet files (default: skip files that exist)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        files = sorted(input_path.glob("*.dbn.zst"))
        out_dir = Path(args.out_dir) if args.out_dir else input_path
    elif input_path.is_file():
        files = [input_path]
        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    else:
        raise SystemExit(f"Not a file or directory: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(files)} file(s) -> {out_dir}")
    for f in files:
        dest = out_dir / f.name.replace(".dbn.zst", ".parquet")
        if dest.exists() and not args.overwrite:
            print(f"  {f.name} ... skipped (exists)")
            continue
        print(f"  {f.name} ... ", end="", flush=True)
        convert_file(f, out_dir, args.price_type, args.overwrite)
        print(f"done -> {dest.name}")

    print("All done.")


if __name__ == "__main__":
    main()
