from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from html import escape
from os import PathLike
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

POLARS_ENGINES = {"auto", "streaming", "gpu"}
SERIES_KINDS = {"level", "event"}


@dataclass
class Series:
    """A plotted y series within a Layer.

    If ``y`` is a Polars expression, ``name`` is required and is used as the
    temporary projected column name. DataSource inputs use ``frame(select=False)``,
    so fitted transforms are applied; pass a raw LazyFrame or transform-free
    DataSource to inspect raw data.
    """

    y: str | pl.Expr
    kind: str = "level"
    name: str | None = None
    secondary_y: bool = False
    color: str | None = None
    line_width: float = 1.4
    line_shape: str = "hv"
    point_symbol: str = "cross"
    point_size: int = 4
    marker_symbol: str = "circle"
    marker_color: str | None = None
    marker_size: int = 8
    label_cols: Sequence[str] = ()
    max_points: int | None = None
    resample: str | None = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in SERIES_KINDS:
            raise ValueError(f"kind must be one of: {sorted(SERIES_KINDS)}")
        self.label_cols = tuple(self.label_cols)
        if isinstance(self.y, pl.Expr) and not self.name:
            raise ValueError("Series.name is required when y is a Polars expression")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be non-negative or None")

    @property
    def trace_name(self) -> str:
        if self.name is not None:
            return self.name
        if isinstance(self.y, str):
            return self.y
        raise ValueError("Series.name is required when y is a Polars expression")

    @property
    def y_col(self) -> str:
        if isinstance(self.y, str):
            return self.y
        return self.trace_name


@dataclass
class Layer:
    source: Any
    series: list[Series]
    x: str = "ts_event"
    filters: tuple[pl.Expr, ...] = ()
    label_cols: Sequence[str] = ()
    sort_x: bool = True

    def __post_init__(self) -> None:
        self.series = list(self.series)
        self.filters = tuple(self.filters)
        self.label_cols = tuple(self.label_cols)
        if not self.series:
            raise ValueError("Layer.series must contain at least one Series")


@dataclass
class Panel:
    layers: list[Layer]
    title: str | None = None
    height: float = 1.0

    def __post_init__(self) -> None:
        self.layers = list(self.layers)
        if not self.layers:
            raise ValueError("Panel.layers must contain at least one Layer")
        if self.height <= 0:
            raise ValueError("Panel.height must be positive")

    @classmethod
    def from_source(
        cls,
        source: Any,
        series: list[Series],
        *,
        x: str = "ts_event",
        filters: tuple[pl.Expr, ...] = (),
        label_cols: Sequence[str] = (),
        sort_x: bool = True,
        title: str | None = None,
        height: float = 1.0,
    ) -> "Panel":
        return cls(
            layers=[
                Layer(
                    source=source,
                    series=series,
                    x=x,
                    filters=filters,
                    label_cols=label_cols,
                    sort_x=sort_x,
                )
            ],
            title=title,
            height=height,
        )


@dataclass
class _LayerData:
    panel_idx: int
    layer: Layer
    df: pl.DataFrame
    x_dtype: Any
    is_datetime_x: bool


@dataclass
class _LiveSeries:
    name: str
    kind: str
    x: list[Any]
    y: list[Any]


@dataclass
class DataViewer:
    panels: list[Panel]
    shared_x: bool = True
    timezone: str | None = None
    spikemode: str = "across"
    spikesnap: str = "cursor"
    hovermode: str = "x"
    max_points: int | None = 200_000
    resample: str = "auto"
    vertical_spacing: float = 0.03
    height: int | None = None
    template: str = "plotly_white"
    polars_engine: str = "streaming"

    def __post_init__(self) -> None:
        self.panels = list(self.panels)
        if not self.panels:
            raise ValueError("DataViewer.panels must contain at least one Panel")
        self.polars_engine = self.polars_engine.lower()
        if self.polars_engine not in POLARS_ENGINES:
            raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be non-negative or None")
        if self.timezone is not None:
            _zoneinfo(self.timezone)

    def figure(
        self, *, start: Any = None, end: Any = None, live: bool = False
    ) -> "Any":
        """Build a Plotly Figure, or FigureWidget when ``live=True``."""

        go, make_subplots = _plotly()
        collected = self._collect_layers(start=start, end=end)
        fig, live_series = self._build_figure(go, make_subplots, collected)
        if not live:
            return fig
        widget = go.FigureWidget(fig)
        _attach_live_hover(widget, live_series)
        return widget

    def show(self, *, start: Any = None, end: Any = None, live: bool = False, **kw) -> None:
        fig = self.figure(start=start, end=end, live=live)
        fig.show(**kw)

    def to_html(self, path: str | PathLike[str], *, start: Any = None, end: Any = None) -> None:
        fig = self.figure(start=start, end=end, live=False)
        fig.write_html(str(path))

    def _collect_layers(self, *, start: Any, end: Any) -> list[_LayerData]:
        out = []
        for panel_idx, panel in enumerate(self.panels):
            for layer in panel.layers:
                lf = _source_to_lazy(layer.source)
                schema = lf.collect_schema()
                x_dtype = _schema_dtype(schema, layer.x)
                is_datetime_x = _is_datetime_dtype(x_dtype)

                lf = self._filter_layer(lf, layer, x_dtype, start, end)
                lf = _add_expression_series(lf, layer)

                needed = _needed_columns(layer)
                lf = lf.select(needed)
                if self.timezone is not None and is_datetime_x:
                    lf = lf.with_columns(_display_time_expr(layer.x, x_dtype, self.timezone))

                df = lf.collect(engine=self.polars_engine)
                if layer.sort_x and df.height:
                    df = df.sort(layer.x)
                out.append(
                    _LayerData(
                        panel_idx=panel_idx,
                        layer=layer,
                        df=df,
                        x_dtype=x_dtype,
                        is_datetime_x=is_datetime_x,
                    )
                )
        return out

    def _filter_layer(
        self,
        lf: pl.LazyFrame,
        layer: Layer,
        x_dtype: Any,
        start: Any,
        end: Any,
    ) -> pl.LazyFrame:
        predicates = list(layer.filters)
        start_value = _coerce_bound(start, x_dtype, self.timezone)
        end_value = _coerce_bound(end, x_dtype, self.timezone)
        if start_value is not None:
            predicates.append(pl.col(layer.x) >= start_value)
        if end_value is not None:
            predicates.append(pl.col(layer.x) <= end_value)
        if not predicates:
            return lf
        mask = predicates[0]
        for predicate in predicates[1:]:
            mask = mask & predicate
        return lf.filter(mask)

    def _build_figure(
        self,
        go: Any,
        make_subplots: Any,
        collected: list[_LayerData],
    ) -> tuple[Any, list[_LiveSeries]]:
        row_heights = _normalize([panel.height for panel in self.panels])
        specs = [
            [
                {
                    "secondary_y": any(
                        series.secondary_y
                        for layer in panel.layers
                        for series in layer.series
                    )
                }
            ]
            for panel in self.panels
        ]
        titles = [panel.title or "" for panel in self.panels]

        fig = make_subplots(
            rows=len(self.panels),
            cols=1,
            shared_xaxes=self.shared_x,
            vertical_spacing=self.vertical_spacing,
            row_heights=row_heights,
            subplot_titles=titles if any(titles) else None,
            specs=specs,
        )

        live_series = []
        row_has_secondary = [row[0]["secondary_y"] for row in specs]
        for item in collected:
            row = item.panel_idx + 1
            for series in item.layer.series:
                trace, live = self._trace_for_series(go, item, series)
                if row_has_secondary[item.panel_idx]:
                    fig.add_trace(
                        trace,
                        row=row,
                        col=1,
                        secondary_y=series.secondary_y,
                    )
                else:
                    fig.add_trace(trace, row=row, col=1)
                live_series.append(live)

        fig.update_xaxes(
            showspikes=True,
            spikemode=self.spikemode,
            spikesnap=self.spikesnap,
            spikethickness=1,
            spikedash="dot",
        )
        layout = {
            "hovermode": self.hovermode,
            "template": self.template,
        }
        if self.height is not None:
            layout["height"] = self.height
        fig.update_layout(**layout)
        return fig, live_series

    def _trace_for_series(
        self, go: Any, item: _LayerData, series: Series
    ) -> tuple[Any, _LiveSeries]:
        df = item.df
        label_cols = _ordered_unique([*item.layer.label_cols, *series.label_cols])
        x_values = df.get_column(item.layer.x).to_list() if df.height else []
        y_values = df.get_column(series.y_col).to_list() if df.height else []
        indices = _resample_indices(
            series.kind,
            y_values,
            _series_max_points(series, self.max_points),
            series.resample if series.resample is not None else self.resample,
        )
        x_values = _take(x_values, indices)
        y_values = _take(y_values, indices)
        customdata = _customdata(df, label_cols, indices) if label_cols else None
        hovertemplate = _hovertemplate(series.trace_name, label_cols, item.is_datetime_x)

        common = {
            "x": x_values,
            "y": y_values,
            "name": series.trace_name,
            "customdata": customdata,
            "hovertemplate": hovertemplate,
        }
        if series.kind == "level":
            trace = go.Scatter(
                mode="lines+markers",
                line=_drop_none(
                    {
                        "shape": series.line_shape,
                        "color": series.color,
                        "width": series.line_width,
                    }
                ),
                marker=_drop_none(
                    {
                        "symbol": series.point_symbol,
                        "size": series.point_size,
                        "color": series.color,
                    }
                ),
                **_drop_none(common),
            )
        else:
            trace = go.Scatter(
                mode="markers",
                marker=_drop_none(
                    {
                        "symbol": series.marker_symbol,
                        "color": series.marker_color,
                        "size": series.marker_size,
                    }
                ),
                **_drop_none(common),
            )
        live = _LiveSeries(series.trace_name, series.kind, x_values, y_values)
        return trace, live


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "DataViewer requires plotly for figure(), show(), and to_html()."
        ) from exc
    return go, make_subplots


def _source_to_lazy(source: Any) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if hasattr(source, "frame"):
        return source.frame(select=False)
    raise TypeError("Layer.source must be a DataSource, LazyFrame, or DataFrame")


def _schema_dtype(schema: Any, col: str) -> Any:
    names = schema.names() if hasattr(schema, "names") else list(schema)
    if col not in names:
        raise ValueError(f"x column {col!r} not found in layer source")
    return schema[col]


def _add_expression_series(lf: pl.LazyFrame, layer: Layer) -> pl.LazyFrame:
    exprs = []
    for series in layer.series:
        if isinstance(series.y, pl.Expr):
            exprs.append(series.y.alias(series.y_col))
    return lf.with_columns(exprs) if exprs else lf


def _needed_columns(layer: Layer) -> list[str]:
    cols = [layer.x]
    for series in layer.series:
        cols.append(series.y_col if isinstance(series.y, pl.Expr) else series.y)
        cols.extend(layer.label_cols)
        cols.extend(series.label_cols)
    return _ordered_unique(cols)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    return [float(value) / total for value in values]


def _series_max_points(series: Series, default: int | None) -> int | None:
    return default if series.max_points is None else series.max_points


def _resample_indices(
    kind: str,
    y_values: Sequence[Any],
    max_points: int | None,
    resample: str | None,
) -> list[int]:
    n = len(y_values)
    if n == 0:
        return []
    if max_points is None or resample == "none" or n <= max_points:
        return list(range(n))
    if max_points <= 0:
        return []
    if kind == "level":
        return _level_indices(y_values, max_points)
    return _uniform_indices(n, max_points)


def _level_indices(y_values: Sequence[Any], max_points: int) -> list[int]:
    n = len(y_values)
    indices = [0]
    for idx in range(1, n):
        if not _same_value(y_values[idx], y_values[idx - 1]):
            indices.append(idx)
    if indices[-1] != n - 1:
        indices.append(n - 1)
    if len(indices) <= max_points:
        return indices
    if max_points == 1:
        return [indices[-1]]
    return _bucket_last(indices, max_points)


def _bucket_last(indices: Sequence[int], max_points: int) -> list[int]:
    if max_points <= 1:
        return [indices[-1]]
    rest = indices[1:]
    buckets = max_points - 1
    edges = np.linspace(0, len(rest), buckets + 1, dtype=int)
    selected = [indices[0]]
    for left, right in zip(edges[:-1], edges[1:]):
        if right > left:
            selected.append(rest[right - 1])
    return _ordered_unique_int(selected)[:max_points]


def _uniform_indices(n: int, max_points: int) -> list[int]:
    if max_points <= 0:
        return []
    if max_points >= n:
        return list(range(n))
    return _ordered_unique_int(np.linspace(0, n - 1, max_points, dtype=int).tolist())


def _ordered_unique_int(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _same_value(left: Any, right: Any) -> bool:
    try:
        if left != left and right != right:
            return True
    except Exception:
        pass
    try:
        return bool(left == right)
    except Exception:
        return False


def _take(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[idx] for idx in indices]


def _customdata(
    df: pl.DataFrame,
    label_cols: Sequence[str],
    indices: Sequence[int],
) -> np.ndarray:
    cols = [_take(df.get_column(col).to_list(), indices) for col in label_cols]
    return np.array(cols, dtype=object).T


def _hovertemplate(name: str, label_cols: Sequence[str], is_datetime_x: bool) -> str:
    x_fmt = "%{x|%H:%M:%S.%f}" if is_datetime_x else "%{x}"
    parts = [f"<b>{escape(name)}</b>", f"x={x_fmt}", "y=%{y}"]
    for idx, col in enumerate(label_cols):
        parts.append(f"{escape(col)}=%{{customdata[{idx}]}}")
    return "<br>".join(parts) + f"<extra>{escape(name)}</extra>"


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _is_datetime_dtype(dtype: Any) -> bool:
    return _base_type(dtype) == pl.Datetime


def _is_date_dtype(dtype: Any) -> bool:
    return _base_type(dtype) == pl.Date


def _base_type(dtype: Any) -> Any:
    base_type = getattr(dtype, "base_type", None)
    return base_type() if callable(base_type) else dtype


def _dtype_time_zone(dtype: Any) -> str | None:
    return getattr(dtype, "time_zone", None)


def _zoneinfo(name: str) -> timezone | ZoneInfo:
    return timezone.utc if name.upper() == "UTC" else ZoneInfo(name)


def _coerce_bound(value: Any, x_dtype: Any, display_tz: str | None) -> Any:
    if value is None:
        return None
    if _is_datetime_dtype(x_dtype):
        return _coerce_datetime_bound(value, x_dtype, display_tz)
    if _is_date_dtype(x_dtype):
        dt = _parse_datetime_like(value)
        return dt.date() if isinstance(dt, datetime) else dt
    return value


def _coerce_datetime_bound(value: Any, x_dtype: Any, display_tz: str | None) -> datetime:
    dt = _parse_datetime_like(value)
    if not isinstance(dt, datetime):
        raise TypeError(f"cannot coerce {value!r} to a datetime bound")

    source_tz = _dtype_time_zone(x_dtype)
    source_zone = _zoneinfo(source_tz) if source_tz is not None else None

    if dt.tzinfo is None:
        if display_tz is not None:
            dt = dt.replace(tzinfo=_zoneinfo(display_tz))
            if source_zone is not None:
                return dt.astimezone(source_zone)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        if source_zone is not None:
            return dt.replace(tzinfo=source_zone)
        return dt

    if source_zone is not None:
        return dt.astimezone(source_zone)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_datetime_like(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="us")
        return _parse_datetime_string(text)
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime()
    if isinstance(value, str):
        return _parse_datetime_string(value)
    return value


def _parse_datetime_string(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _display_time_expr(x: str, x_dtype: Any, display_tz: str) -> pl.Expr:
    expr = pl.col(x)
    if _dtype_time_zone(x_dtype) is None:
        expr = expr.dt.replace_time_zone("UTC")
    return expr.dt.convert_time_zone(display_tz).alias(x)


def _attach_live_hover(fig: Any, live_series: list[_LiveSeries]) -> None:
    if not live_series:
        return
    fig.add_annotation(
        x=1.0,
        y=1.0,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font={"size": 12},
        text="",
    )
    annotation_idx = len(fig.layout.annotations) - 1

    def on_hover(trace: Any, points: Any, state: Any) -> None:
        hovered = _hover_x(points, trace)
        if hovered is None:
            return
        lines = []
        for item in live_series:
            idx = _live_index(item, hovered)
            if idx is not None:
                lines.append(f"{escape(item.name)}: {escape(str(item.y[idx]))}")
        with fig.batch_update():
            fig.layout.annotations[annotation_idx].text = "<br>".join(lines)

    for trace in fig.data:
        trace.on_hover(on_hover)


def _hover_x(points: Any, trace: Any) -> Any:
    xs = getattr(points, "xs", None)
    if xs:
        return xs[0]
    point_inds = getattr(points, "point_inds", None)
    if point_inds:
        return trace.x[point_inds[0]]
    return None


def _live_index(item: _LiveSeries, hovered: Any) -> int | None:
    if not item.x:
        return None
    value = _coerce_live_value(hovered, item.x[0])
    try:
        pos = int(np.searchsorted(item.x, value, side="right"))
    except Exception:
        return None
    if item.kind == "level":
        idx = pos - 1
        return idx if 0 <= idx < len(item.x) else None
    candidates = []
    if 0 <= pos < len(item.x):
        candidates.append(pos)
    if 0 <= pos - 1 < len(item.x):
        candidates.append(pos - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: _distance(item.x[idx], value))


def _coerce_live_value(value: Any, sample: Any) -> Any:
    if isinstance(sample, datetime) and isinstance(value, str):
        parsed = _parse_datetime_string(value)
        if sample.tzinfo is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=sample.tzinfo)
        if sample.tzinfo is None and parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return value


def _distance(left: Any, right: Any) -> float:
    try:
        value = abs(left - right)
        return value.total_seconds() if hasattr(value, "total_seconds") else float(value)
    except Exception:
        return 0.0 if left == right else float("inf")


__all__ = ["DataViewer", "Layer", "Panel", "Series"]
