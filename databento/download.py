import argparse

import databento as db
from datetime import date, timedelta
from pathlib import Path

from databento_auth import DEFAULT_API_KEY_FILE, read_databento_api_key

out_dir = Path("data/databento_glbx_mdp3_mbo_full_utc_day")
out_dir.mkdir(parents=True, exist_ok=True)

DATASET = "GLBX.MDP3"
SCHEMA = "mbo"
STYPE_IN = "raw_symbol"

FULL_DAY_START_UTC = "00:00:00Z"
FULL_DAY_END_UTC = "00:00:00Z"
EXISTING_START_UTC = "13:15:00Z"
EXISTING_END_UTC = "20:05:00Z"


def next_day(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description="Download configured Databento MBO ranges")
    parser.add_argument("--start-date", help="First request date to include, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Last request date to include, YYYY-MM-DD")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help="Encrypted GPG file containing the Databento API key (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["full-day", "gap", "existing"],
        default="full-day",
        help=(
            "Range to download: full-day downloads 00:00Z to next-day 00:00Z; "
            "gap downloads only 00:00Z-13:15Z and 20:05Z-next-day 00:00Z; "
            "existing downloads the original 13:15Z-20:05Z window"
        ),
    )
    parser.add_argument(
        "--cost-cap",
        type=float,
        help="Maximum estimated Databento cost in USD before any pending downloads start",
    )
    return parser.parse_args()


def validate_date(value: str | None) -> str | None:
    if value is None:
        return None
    date.fromisoformat(value)
    return value


def data_ranges(day: str, mode: str) -> list[tuple[str, str, str]]:
    if mode == "full-day":
        return [
            (
                "full_day",
                f"{day}T{FULL_DAY_START_UTC}",
                f"{next_day(day)}T{FULL_DAY_END_UTC}",
            )
        ]
    if mode == "existing":
        return [
            (
                "existing",
                f"{day}T{EXISTING_START_UTC}",
                f"{day}T{EXISTING_END_UTC}",
            )
        ]
    return [
        (
            "gap_before",
            f"{day}T{FULL_DAY_START_UTC}",
            f"{day}T{EXISTING_START_UTC}",
        ),
        (
            "gap_after",
            f"{day}T{EXISTING_END_UTC}",
            f"{next_day(day)}T{FULL_DAY_END_UTC}",
        ),
    ]


def download_jobs(requests: list[tuple[str, str, str]], mode: str):
    for symbol, day, tag in requests:
        for label, start, end in data_ranges(day, mode):
            suffix = "" if mode == "full-day" else f"_{label}"
            out_path = out_dir / f"{symbol}_{day}_{tag}{suffix}.dbn.zst"
            yield symbol, day, tag, label, start, end, out_path


def check_cost_cap(client, jobs, cap: float | None) -> None:
    if cap is None:
        return
    if cap < 0:
        raise SystemExit("--cost-cap must be non-negative")

    total = 0.0
    for symbol, day, tag, label, start, end, out_path in jobs:
        if out_path.exists():
            continue
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=symbol,
            schema=SCHEMA,
            stype_in=STYPE_IN,
            start=start,
            end=end,
        )
        total += cost
        print(f"estimated {symbol} {day} {tag} {label} {start} -> {end}: ${cost:.4f}")

    print(f"TOTAL ESTIMATED PENDING COST: ${total:.2f}")
    if total > cap:
        raise SystemExit(
            f"Estimated pending cost ${total:.2f} exceeds --cost-cap ${cap:.2f}; "
            "no files downloaded"
        )

es_normal = [
    "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17",
    "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24",
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
    "2026-05-11", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-21", "2026-05-22",
]

nq_transfer = [
    "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24",
    "2026-05-11", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-21", "2026-05-22",
]

stress_days = [
    "2026-04-10",
    "2026-04-29",
    "2026-05-08",
    "2026-05-12",
    "2026-05-20",
]

holiday_days = [
    "2026-05-25",
]

requests = []

for d in es_normal:
    requests.append(("ESM6", d, "normal_es"))

for d in nq_transfer:
    requests.append(("NQM6", d, "transfer_nq"))

for d in stress_days:
    requests.append(("ESM6", d, "stress_es"))
    requests.append(("NQM6", d, "stress_nq"))

for d in holiday_days:
    requests.append(("ESM6", d, "holiday_es"))
    requests.append(("NQM6", d, "holiday_nq"))

extra_es_normal = [
    "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27",
    "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02",
    "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
    "2026-04-27", "2026-04-28", "2026-04-30", "2026-05-01",
    "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
]

extra_nq_transfer = [
    "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02",
    "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
    "2026-04-27", "2026-04-28", "2026-04-30", "2026-05-01",
    "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
]

for d in extra_es_normal:
    requests.append(("ESM6", d, "extra_normal_es"))

for d in extra_nq_transfer:
    requests.append(("NQM6", d, "extra_transfer_nq"))


args = parse_args()
start_date = validate_date(args.start_date)
end_date = validate_date(args.end_date)
if start_date and end_date and start_date > end_date:
    raise SystemExit("--start-date must be on or before --end-date")
if start_date:
    requests = [r for r in requests if r[1] >= start_date]
if end_date:
    requests = [r for r in requests if r[1] <= end_date]
if not requests:
    raise SystemExit("No configured requests match the selected date range")

client = db.Historical(read_databento_api_key(args.api_key_file))
jobs = list(download_jobs(requests, args.mode))
check_cost_cap(client, jobs, args.cost_cap)

for symbol, day, tag, label, start, end, out_path in jobs:
    if out_path.exists():
        print("skipped", out_path, "(exists)")
        continue

    client.timeseries.get_range(
        dataset=DATASET,
        symbols=symbol,
        schema=SCHEMA,
        stype_in=STYPE_IN,
        start=start,
        end=end,
        path=str(out_path),
    )

    print("saved", out_path)
