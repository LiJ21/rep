from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import torch
from torch import nn

from tools.data import DateFrame, LoadHint, Raw


ROOT = Path(__file__).resolve().parents[1]
L2_PATH = str(
    ROOT
    / "data/orderbook_l2_parquet/{prod}M6_{d}_{tag}_{prod_s}_full_day_l2_d5.parquet"
)
RETURN_PATH = str(
    ROOT
    / "data/orderbook_feature_return_parquet/"
    "{prod}M6_{d}_{tag}_{prod_s}_full_day_l2_d5_features_return.parquet"
)
TARGET = "forward_mid_return_bps"
ALIGNMENT_KEYS = (
    "ts_event",
    "row_nr",
    "sequence",
    "publisher_id",
    "instrument_id",
)
WINDOW_GROUP_COLS = ("publisher_id", "instrument_id")
ALIGNMENT_COL = "__deeplob_aligned"
ENDPOINT_COL = "__window_endpoint"
DEEPLOB_FEATURES = tuple(
    f"{field}_{level}"
    for level in range(5)
    for field in (
        "ask_px_offset",
        "ask_sz_log1p",
        "bid_px_offset",
        "bid_sz_log1p",
    )
)


@dataclass(frozen=True)
class DeepLOBLoader:
    """Aligned depth-5 input and stored-label loader on an occupied RTH clock."""

    alignment_cols: ClassVar[tuple[str, ...]] = (ALIGNMENT_COL,)
    window_group_cols: ClassVar[tuple[str, ...]] = WINDOW_GROUP_COLS
    endpoint_col: ClassVar[str] = ENDPOINT_COL

    prod: str = "ES"
    l2_path: str = L2_PATH
    return_path: str = RETURN_PATH
    target: str = TARGET
    levels: int = 5
    bar: str = "250ms"
    tick_size: float = 0.25
    price_scale: float = 1e9
    timezone: str = "America/New_York"
    rth_open: dt.time = dt.time(9, 30)
    rth_close: dt.time = dt.time(16)
    label_horizon_s: int = 60

    def __post_init__(self) -> None:
        if self.levels != 5:
            raise ValueError("DeepLOB v1 requires exactly five levels")
        if self.tick_size <= 0 or self.price_scale <= 0:
            raise ValueError("tick_size and price_scale must be positive")
        if self.label_horizon_s < 0:
            raise ValueError("label_horizon_s must be nonnegative")
        if self.rth_open >= self.rth_close:
            raise ValueError("rth_open must precede rth_close")

    @property
    def features(self) -> tuple[str, ...]:
        return DEEPLOB_FEATURES

    @property
    def feature_exprs(self) -> dict[str, pl.Expr]:
        mid = (
            pl.col("ask_px_0").cast(pl.Float64)
            + pl.col("bid_px_0").cast(pl.Float64)
        ) / 2
        tick = self.tick_size * self.price_scale
        return {
            name: (
                ((pl.col(f"{side}_px_{level}").cast(pl.Float64) - mid) / tick)
                if kind == "px_offset"
                else pl.col(f"{side}_sz_{level}").cast(pl.Float64).log1p()
            ).alias(name)
            for level in range(self.levels)
            for side, kind, name in (
                ("ask", "px_offset", f"ask_px_offset_{level}"),
                ("ask", "sz_log1p", f"ask_sz_log1p_{level}"),
                ("bid", "px_offset", f"bid_px_offset_{level}"),
                ("bid", "sz_log1p", f"bid_sz_log1p_{level}"),
            )
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "product": self.prod.upper(),
            "clock": f"occupied_bucket_last:{self.bar}",
            "session": f"{self.rth_open}-{self.rth_close}:{self.timezone}",
            "label": self.target,
            "label_horizon_s": self.label_horizon_s,
            "features": list(self.features),
            "alignment_keys": list(ALIGNMENT_KEYS),
            "window_group_cols": list(WINDOW_GROUP_COLS),
            "endpoint_col": ENDPOINT_COL,
        }

    def to_config(self) -> dict[str, Any]:
        return {
            "prod": self.prod,
            "l2_path": self.l2_path,
            "return_path": self.return_path,
            "target": self.target,
            "levels": self.levels,
            "bar": self.bar,
            "tick_size": self.tick_size,
            "price_scale": self.price_scale,
            "timezone": self.timezone,
            "rth_open": self.rth_open.isoformat(),
            "rth_close": self.rth_close.isoformat(),
            "label_horizon_s": self.label_horizon_s,
        }

    def feature_files(self, dates: Sequence[str]) -> list[Path]:
        files: list[Path] = []
        for date in dates:
            l2, l2_tag = Raw.resolve_path(date, self.prod.upper(), self.l2_path)
            ret, ret_tag = Raw.resolve_path(
                date, self.prod.upper(), self.return_path
            )
            if l2_tag != ret_tag:
                raise ValueError(
                    f"regime mismatch for {date}: L2={l2_tag!r}, returns={ret_tag!r}"
                )
            files.extend((Path(l2), Path(ret)))
        return files

    def __call__(
        self, dates: str | Sequence[str], hint: LoadHint | None = None
    ) -> list[DateFrame]:
        del hint
        values = [dates] if isinstance(dates, str) else dates
        return [self.load_date(date) for date in values]

    def iter_date_frames(
        self, dates: str | Sequence[str], hint: LoadHint | None = None
    ):
        del hint
        values = [dates] if isinstance(dates, str) else dates
        for date in values:
            yield self.load_date(date)

    def load_date(self, date: str) -> DateFrame:
        l2_path, nature = Raw.resolve_path(date, self.prod.upper(), self.l2_path)
        return_file, return_nature = Raw.resolve_path(
            date, self.prod.upper(), self.return_path
        )
        if nature != return_nature:
            raise ValueError(
                f"regime mismatch for {date}: L2={nature!r}, "
                f"returns={return_nature!r}"
            )

        raw_cols = [
            *ALIGNMENT_KEYS,
            *(
                f"{side}_{field}_{level}"
                for level in range(self.levels)
                for side in ("ask", "bid")
                for field in ("px", "sz")
            ),
        ]
        labels = pl.scan_parquet(return_file).select(
            self.target,
            *(pl.col(key).alias(f"__label_{key}") for key in ALIGNMENT_KEYS),
        )
        lf = pl.concat(
            [pl.scan_parquet(l2_path).select(raw_cols), labels], how="horizontal"
        )
        aligned = pl.all_horizontal(
            *(pl.col(key) == pl.col(f"__label_{key}") for key in ALIGNMENT_KEYS)
        ).fill_null(False)
        lf = lf.with_columns(
            aligned.all().over(pl.lit(0)).alias(ALIGNMENT_COL)
        )
        local = pl.col("ts_event").dt.convert_time_zone(self.timezone).dt.time()
        lf = (
            lf.filter((local >= self.rth_open) & (local < self.rth_close))
            .with_columns(pl.col("ts_event").dt.truncate(self.bar).alias("__bucket"))
            .group_by(*WINDOW_GROUP_COLS, "__bucket", maintain_order=True)
            .agg(pl.all().last())
        )
        valid = pl.all_horizontal(
            *(
                (pl.col(f"{side}_px_{level}") > 0)
                & (
                    pl.col(f"{side}_px_{level}")
                    != pl.lit(2**63 - 1, dtype=pl.Int64)
                )
                for level in range(self.levels)
                for side in ("ask", "bid")
            ),
            pl.col("ask_px_0") > pl.col("bid_px_0"),
        )
        endpoint = (
            (
                (pl.col("ts_event") + pl.duration(seconds=self.label_horizon_s))
                .dt.convert_time_zone(self.timezone)
                .dt.time()
                <= self.rth_close
            )
            & pl.col(self.target).is_not_null()
            & pl.col(self.target).is_finite()
        )
        return DateFrame(
            date=date,
            nature=nature,
            lf=lf.filter(valid).with_columns(
                endpoint.alias(ENDPOINT_COL),
                *self.feature_exprs.values(),
            ).select(
                *ALIGNMENT_KEYS,
                self.target,
                *self.features,
                ALIGNMENT_COL,
                ENDPOINT_COL,
            ),
        )

    @staticmethod
    def assert_aligned(df: pl.DataFrame, date: str | None = None) -> None:
        if ALIGNMENT_COL not in df:
            raise ValueError(f"missing required alignment column {ALIGNMENT_COL!r}")
        bad = df.get_column(ALIGNMENT_COL).fill_null(False).not_()
        if bad.any():
            row = int(bad.arg_true()[0])
            suffix = f" for {date}" if date else ""
            raise ValueError(f"L2/return ordered-key mismatch{suffix} at row {row}")


class DeepLOB(nn.Module):
    """Five-level DeepLOB CNN-inception-LSTM regressor."""

    def __init__(
        self,
        channels: int = 32,
        inception_channels: int = 64,
        hidden_size: int = 64,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        def conv(in_channels: int, width: int, stride: int = 1) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    channels,
                    (1, width),
                    stride=(1, stride),
                ),
                nn.LeakyReLU(negative_slope),
                nn.BatchNorm2d(channels),
            ]
            for _ in range(2):
                layers.extend(
                    (
                        nn.Conv2d(channels, channels, (4, 1)),
                        nn.LeakyReLU(negative_slope),
                        nn.BatchNorm2d(channels),
                    )
                )
            return nn.Sequential(*layers)

        self.convs = nn.Sequential(
            conv(1, 2, 2),
            conv(channels, 2, 2),
            conv(channels, 5),
        )
        self.inception = nn.ModuleList(
            tuple(
                nn.Sequential(
                    nn.Conv2d(channels, inception_channels, 1),
                    nn.LeakyReLU(negative_slope),
                    nn.BatchNorm2d(inception_channels),
                    nn.Conv2d(
                        inception_channels,
                        inception_channels,
                        (kernel, 1),
                        padding=(kernel // 2, 0),
                    ),
                    nn.LeakyReLU(negative_slope),
                    nn.BatchNorm2d(inception_channels),
                )
                for kernel in (3, 5)
            )
            + (
                nn.Sequential(
                    nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
                    nn.Conv2d(channels, inception_channels, 1),
                    nn.LeakyReLU(negative_slope),
                    nn.BatchNorm2d(inception_channels),
                ),
            )
        )
        self.lstm = nn.LSTM(3 * inception_channels, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 20:
            raise ValueError(f"expected (B,T,20), got {tuple(x.shape)}")
        x = self.convs(x.unsqueeze(1))
        x = torch.cat([branch(x) for branch in self.inception], dim=1)
        x = x.squeeze(-1).transpose(1, 2)
        return self.head(self.lstm(x)[0][:, -1])


def build_deeplob(params: Mapping[str, Any] | None = None) -> DeepLOB:
    params = params or {}
    torch.manual_seed(int(params.get("seed", 0)))
    return DeepLOB(
        channels=int(params.get("channels", 32)),
        inception_channels=int(params.get("inception_channels", 64)),
        hidden_size=int(params.get("hidden_size", 64)),
        negative_slope=float(params.get("negative_slope", 0.01)),
    )
