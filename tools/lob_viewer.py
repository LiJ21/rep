from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from tools.viewer import _coerce_bound, _schema_dtype, _source_to_lazy, plot_timeseries

BUY_SIDE = 0
SELL_SIDE = 1
PRICE_SCALE = 1_000_000_000
INVALID_PRICE = 9_223_372_036_854_775_807

BID_COLORS = ("#00875a", "#00a876", "#29b889", "#55c59c", "#82d2b2")
ASK_COLORS = ("#d72638", "#ef4444", "#f97316", "#fb923c", "#fdba74")
DEFAULT_L3_ACTIONS = {
    "trade": "T",
    "add": "A",
    "modify": "M",
    "cancel": "C",
}
EVENT_OVERFLOW_MODES = {"hide", "sample"}
PREDICTION_COLORS = ("#7c3aed", "#2563eb", "#0891b2", "#059669", "#ca8a04")
RETURN_COLOR = "#f43f5e"
DEFAULT_SIGNAL_LABEL_COLS = ("row_nr", "instrument_id")


def plot_order_book(
    source: Any,
    *,
    x: str = "ts_event",
    depth: int = 5,
    bid_price_cols: Sequence[str] | None = None,
    ask_price_cols: Sequence[str] | None = None,
    bid_size_cols: Sequence[str] | None = None,
    ask_size_cols: Sequence[str] | None = None,
    bid_count_cols: Sequence[str] | None = None,
    ask_count_cols: Sequence[str] | None = None,
    trade_price_col: str = "trade_px",
    trade_size_col: str = "trade_sz",
    trade_side_col: str = "trade_side",
    buy_value: Any = BUY_SIDE,
    sell_value: Any = SELL_SIDE,
    l3_source: Any | None = None,
    l3_x: str | None = None,
    l3_price_col: str = "price",
    l3_size_col: str = "size",
    l3_side_col: str = "side",
    l3_action_col: str = "action",
    l3_flags_col: str | None = "flags",
    l3_bid_value: Any = "B",
    l3_ask_value: Any = "A",
    l3_actions: Mapping[str, Any] | None = None,
    l3_depth_filter: str | None = "levels",
    l3_filters: Sequence[pl.Expr] = (),
    l3_label_cols: Sequence[str] = ("order_id", "sequence"),
    signal_source: Any | None = None,
    signal_x: str | None = None,
    signal_cols: Sequence[str] = (),
    prediction_cols: Sequence[str] = (),
    return_col: str | None = None,
    signal_label_cols: Sequence[str] = DEFAULT_SIGNAL_LABEL_COLS,
    signal_height: float = 0.28,
    signal_title: str | None = "prediction and forward return",
    signal_yaxis_title: str | None = "bps",
    signal_value_label: str | None = "bps",
    prediction_colors: Sequence[str] = PREDICTION_COLORS,
    return_color: str | None = RETURN_COLOR,
    include_trades: bool = True,
    include_sizes: bool = True,
    include_counts: bool = False,
    filters: Sequence[pl.Expr] = (),
    label_cols: Sequence[str] = (),
    start: Any = None,
    end: Any = None,
    timezone: str | None = None,
    product: str | None = None,
    title: str | None = None,
    price_scale: float | None = PRICE_SCALE,
    invalid_price: int | None = INVALID_PRICE,
    live: bool = False,
    return_viewer: bool = False,
    height: int | None = 860,
    max_points: int | None = 200_000,
    event_overflow: str = "hide",
    max_event_points: int | None = None,
    resample: str = "auto",
    template: str = "plotly_white",
    polars_engine: str = "streaming",
    layout: Mapping[str, Any] | None = None,
    bid_colors: Sequence[str] = BID_COLORS,
    ask_colors: Sequence[str] = ASK_COLORS,
    **viewer_kwargs: Any,
) -> Any:
    """Plot an L2 order book with market-data-oriented defaults.

    With ``event_overflow="hide"``, trades or L3 events are plotted in full
    while their aggregate selected-window population fits the event budget;
    otherwise every event trace is omitted and the level lines remain.
    ``event_overflow="sample"`` retains the legacy per-trace sampling behavior.

    A ``signal_source`` adds a shared-x line panel. ``signal_cols`` can contain
    arbitrary feature or return columns; the legacy ``prediction_cols`` and
    ``return_col`` arguments remain available for model-oriented plots. Values
    are plotted unchanged, so choose an axis title appropriate to their units.

    A finite hide-mode budget is tied to the requested window and is therefore
    incompatible with ``return_viewer=True`` while events are active. Use
    ``event_overflow="sample"`` for reuse, or rebuild for each new window.
    """

    if depth <= 0:
        raise ValueError("depth must be positive")
    if price_scale is not None and price_scale <= 0:
        raise ValueError("price_scale must be positive or None")
    event_overflow = event_overflow.lower()
    if event_overflow not in EVENT_OVERFLOW_MODES:
        raise ValueError(
            f"event_overflow must be one of: {sorted(EVENT_OVERFLOW_MODES)}"
        )
    event_budget = max_points if max_event_points is None else max_event_points
    if event_budget is not None and event_budget < 0:
        raise ValueError("max_event_points must be non-negative or None")
    if l3_depth_filter not in {None, "levels", "range"}:
        raise ValueError("l3_depth_filter must be 'levels', 'range', or None")
    signal_cols = _ordered_unique(
        (*_str_tuple(signal_cols), *_str_tuple(prediction_cols))
    )
    signal_label_cols = _str_tuple(signal_label_cols)
    if signal_source is None and (signal_cols or return_col is not None):
        raise ValueError(
            "signal_source is required when signal columns or return_col is provided"
        )
    if signal_source is not None and not signal_cols and return_col is None:
        raise ValueError(
            "signal_source requires at least one prediction column or return_col"
        )
    if signal_source is not None and signal_height <= 0:
        raise ValueError("signal_height must be positive")
    if signal_source is not None:
        signal_label_cols = _available_source_cols(signal_source, signal_label_cols)
    viewer_kwargs.setdefault("hovermode", "x")
    viewer_kwargs.setdefault("x_label", "time")
    viewer_kwargs.setdefault("datetime_x_mode", "relative_ns")
    viewer_kwargs.setdefault("spikesnap", "data")

    bid_px = _level_cols(bid_price_cols, "bid_px", depth)
    ask_px = _level_cols(ask_price_cols, "ask_px", depth)
    bid_sz = _level_cols(bid_size_cols, "bid_sz", depth)
    ask_sz = _level_cols(ask_size_cols, "ask_sz", depth)
    bid_ct = _optional_level_cols(bid_count_cols, "bid_ct", depth, include_counts)
    ask_ct = _optional_level_cols(ask_count_cols, "ask_ct", depth, include_counts)

    l2_filters = _expr_tuple(filters)
    l2_label_cols = _str_tuple(label_cols)
    resolved_l3_actions = DEFAULT_L3_ACTIONS if l3_actions is None else l3_actions
    active_events = (
        (l3_source is not None and bool(resolved_l3_actions))
        or (l3_source is None and include_trades)
    )
    if (
        return_viewer
        and event_overflow == "hide"
        and event_budget is not None
        and active_events
    ):
        raise ValueError(
            "return_viewer=True cannot be used with a finite hide-mode event "
            "budget because the overflow decision is fixed to the initial window; "
            "use event_overflow='sample' or rebuild plot_order_book for each window"
        )
    omitted_event_kind: str | None = None
    if event_overflow == "hide" and event_budget is not None:
        if l3_source is not None and _l3_events_over_budget(
            l3_source,
            x=l3_x or x,
            start=start,
            end=end,
            timezone=timezone,
            filters=l3_filters,
            price_col=l3_price_col,
            size_col=l3_size_col,
            side_col=l3_side_col,
            action_col=l3_action_col,
            bid_value=l3_bid_value,
            ask_value=l3_ask_value,
            actions=resolved_l3_actions,
            invalid_price=invalid_price,
            max_points=event_budget,
            polars_engine=polars_engine,
        ):
            omitted_event_kind = "L3 events"
        elif l3_source is None and include_trades and _l2_trades_over_budget(
            source,
            x=x,
            start=start,
            end=end,
            timezone=timezone,
            filters=l2_filters,
            price_col=trade_price_col,
            size_col=trade_size_col,
            side_col=trade_side_col,
            buy_value=buy_value,
            sell_value=sell_value,
            invalid_price=invalid_price,
            max_points=event_budget,
            polars_engine=polars_engine,
        ):
            omitted_event_kind = "trades"

    events_omitted = omitted_event_kind is not None
    raw_events = event_overflow == "hide" and not events_omitted
    show_l2_depth_points = l3_source is None and not events_omitted
    l2_depth_point_size = 3 if show_l2_depth_points else 0
    price_series = []
    size_series = []
    for level in range(depth):
        label = level + 1
        bid_color = _palette_color(bid_colors, level)
        ask_color = _palette_color(ask_colors, level)
        price_series.extend(
            [
                {
                    "y": _price_expr(
                        bid_px[level],
                        bid_sz[level],
                        price_scale=price_scale,
                        invalid_price=invalid_price,
                    ),
                    "name": f"bid L{label}",
                    "color": bid_color,
                    "point_symbol": "cross",
                    "point_size": l2_depth_point_size,
                    "value_label": "price",
                    "hover": show_l2_depth_points,
                    "label_cols": _level_label_cols(bid_sz[level], bid_ct[level]),
                },
                {
                    "y": _price_expr(
                        ask_px[level],
                        ask_sz[level],
                        price_scale=price_scale,
                        invalid_price=invalid_price,
                    ),
                    "name": f"ask L{label}",
                    "color": ask_color,
                    "point_symbol": "cross",
                    "point_size": l2_depth_point_size,
                    "value_label": "price",
                    "hover": show_l2_depth_points,
                    "label_cols": _level_label_cols(ask_sz[level], ask_ct[level]),
                },
            ]
        )
        size_series.extend(
            [
                {
                    "y": _size_expr(
                        bid_sz[level],
                        bid_px[level],
                        invalid_price=invalid_price,
                    ),
                    "name": f"bid size L{label}",
                    "color": bid_color,
                    "point_symbol": "cross",
                    "point_size": l2_depth_point_size,
                    "value_label": "size",
                    "hover": show_l2_depth_points,
                    "label_cols": _level_label_cols(bid_px[level], bid_ct[level]),
                },
                {
                    "y": _size_expr(
                        ask_sz[level],
                        ask_px[level],
                        invalid_price=invalid_price,
                    ),
                    "name": f"ask size L{label}",
                    "color": ask_color,
                    "point_symbol": "cross",
                    "point_size": l2_depth_point_size,
                    "value_label": "size",
                    "hover": show_l2_depth_points,
                    "label_cols": _level_label_cols(ask_px[level], ask_ct[level]),
                },
            ]
        )

    depth_price_layer = {
        "source": source,
        "y": price_series,
        "filters": l2_filters,
        "label_cols": l2_label_cols,
    }
    price_layers: list[dict[str, Any]] = []
    if include_trades and l3_source is None and not events_omitted:
        valid_trade = _valid_price(
            trade_price_col,
            trade_size_col,
            invalid_price=invalid_price,
        )
        trade_y = _price_expr(
            trade_price_col,
            trade_size_col,
            price_scale=price_scale,
            invalid_price=invalid_price,
        )
        price_layers.extend(
            [
                {
                    "source": source,
                    "events": [
                        {
                            "y": trade_y,
                            "name": "buy trades",
                            "marker_symbol": "triangle-up",
                            "marker_color": "#00c853",
                            "marker_size": 9,
                            "value_label": "price",
                            "label_cols": (trade_size_col, trade_side_col),
                            "resample": "none" if raw_events else None,
                        }
                    ],
                    "filters": _merge_filters(
                        l2_filters,
                        valid_trade & (pl.col(trade_side_col) == buy_value),
                    ),
                    "label_cols": l2_label_cols,
                },
                {
                    "source": source,
                    "events": [
                        {
                            "y": trade_y,
                            "name": "sell trades",
                            "marker_symbol": "triangle-down",
                            "marker_color": "#ff1744",
                            "marker_size": 9,
                            "value_label": "price",
                            "label_cols": (trade_size_col, trade_side_col),
                            "resample": "none" if raw_events else None,
                        }
                    ],
                    "filters": _merge_filters(
                        l2_filters,
                        valid_trade & (pl.col(trade_side_col) == sell_value),
                    ),
                    "label_cols": l2_label_cols,
                },
            ]
        )
    if l3_source is not None and not events_omitted:
        l3_flags_label_col = _optional_source_col(l3_source, l3_flags_col)
        depth_price_filter = _l2_depth_price_filter(
            source,
            mode=l3_depth_filter,
            l3_price_col=l3_price_col,
            x=x,
            start=start,
            end=end,
            timezone=timezone,
            filters=l2_filters,
            bid_price_cols=bid_px,
            ask_price_cols=ask_px,
            bid_size_cols=bid_sz,
            ask_size_cols=ask_sz,
            invalid_price=invalid_price,
            price_scale=price_scale,
            polars_engine=polars_engine,
        )
        price_layers.extend(
            _l3_event_layers(
                source=l3_source,
                x=l3_x or x,
                price_col=l3_price_col,
                size_col=l3_size_col,
                side_col=l3_side_col,
                action_col=l3_action_col,
                bid_value=l3_bid_value,
                ask_value=l3_ask_value,
                actions=resolved_l3_actions,
                filters=l3_filters,
                depth_price_filter=depth_price_filter,
                flags_col=l3_flags_label_col,
                label_cols=l3_label_cols,
                price_scale=price_scale,
                invalid_price=invalid_price,
                event_resample="none" if raw_events else None,
            )
        )
    price_layers.append(depth_price_layer)

    panels = [
        {
            "title": _price_panel_title(
                product,
                (
                    "L3 events"
                    if l3_source is not None
                    else "trades" if include_trades else None
                ),
                omitted_event_kind=omitted_event_kind,
                event_budget=event_budget,
            ),
            "height": 0.68 if include_sizes else 1.0,
            "yaxis_title": "price",
            "layers": price_layers,
        }
    ]
    if include_sizes:
        panels.append(
            {
                "title": "displayed bid/ask size levels",
                "height": 0.32,
                "yaxis_title": "displayed contracts",
                "layers": [
                    {
                        "source": source,
                        "y": size_series,
                        "filters": l2_filters,
                        "label_cols": l2_label_cols,
                    }
                ],
            }
        )

    if signal_source is not None:
        signal_series = []
        for idx, col in enumerate(signal_cols):
            signal_series.append(
                _signal_series(
                    col,
                    color=_palette_color(prediction_colors, idx),
                    value_label=signal_value_label,
                )
            )
        if return_col is not None:
            signal_series.append(
                _signal_series(
                    return_col,
                    color=return_color,
                    value_label=signal_value_label,
                )
            )
        panels.append(
            {
                "title": signal_title,
                "height": signal_height,
                "yaxis_title": signal_yaxis_title,
                "layers": [
                    {
                        "source": signal_source,
                        "x": signal_x or x,
                        "y": signal_series,
                        "label_cols": signal_label_cols,
                    }
                ],
            }
        )

    event_notice = (
        _event_omission_notice(omitted_event_kind, event_budget)
        if omitted_event_kind is not None and event_budget is not None
        else None
    )

    result = plot_timeseries(
        panels=panels,
        x=x,
        start=start,
        end=end,
        live=live,
        return_viewer=return_viewer,
        timezone=timezone,
        height=height,
        max_points=max_points,
        resample=resample,
        template=template,
        polars_engine=polars_engine,
        layout=_layout(
            title,
            product,
            layout,
            event_notice=event_notice,
            omitted_event_kind=omitted_event_kind,
            event_budget=event_budget,
        ),
        **viewer_kwargs,
    )
    if signal_source is not None:
        fig = result[0] if return_viewer else result
        fig.add_hline(
            y=0,
            row=len(panels),
            col=1,
            line_color="#94a3b8",
            line_width=1,
            line_dash="dot",
        )
    return result


def _level_cols(cols: Sequence[str] | None, prefix: str, depth: int) -> list[str]:
    if cols is None:
        return [f"{prefix}_{level}" for level in range(depth)]
    out = list(cols)
    if len(out) < depth:
        raise ValueError(f"{prefix} cols must contain at least depth={depth} columns")
    return out[:depth]


def _optional_level_cols(
    cols: Sequence[str] | None,
    prefix: str,
    depth: int,
    enabled: bool,
) -> list[str | None]:
    if cols is None:
        return [f"{prefix}_{level}" for level in range(depth)] if enabled else [None] * depth
    out = list(cols)
    if len(out) < depth:
        raise ValueError(f"{prefix} cols must contain at least depth={depth} columns")
    return out[:depth]


def _level_label_cols(*cols: str | None) -> tuple[str, ...]:
    out = [col for col in cols if col is not None]
    return tuple(out)


def _optional_source_col(source: Any, col: str | None) -> str | None:
    if col is None:
        return None
    schema = _source_to_lazy(source).collect_schema()
    names = schema.names() if hasattr(schema, "names") else list(schema)
    return col if col in names else None


def _available_source_cols(source: Any, cols: Sequence[str]) -> tuple[str, ...]:
    schema = _source_to_lazy(source).collect_schema()
    names = set(schema.names() if hasattr(schema, "names") else schema)
    return tuple(col for col in cols if col in names)


def _signal_series(
    col: str,
    *,
    color: str | None,
    value_label: str | None,
) -> dict[str, Any]:
    return {
        "y": pl.col(col).cast(pl.Float64),
        "name": col,
        "kind": "line",
        "color": color,
        "line_shape": "linear",
        "point_size": 0,
        "value_label": value_label,
    }


def _l2_trades_over_budget(
    source: Any,
    *,
    x: str,
    start: Any,
    end: Any,
    timezone: str | None,
    filters: Sequence[pl.Expr],
    price_col: str,
    size_col: str,
    side_col: str,
    buy_value: Any,
    sell_value: Any,
    invalid_price: int | None,
    max_points: int,
    polars_engine: str,
) -> bool:
    lf = _windowed_l2_depth_source(
        source,
        x=x,
        start=start,
        end=end,
        timezone=timezone,
        filters=filters,
    )
    eligible = _valid_price(price_col, size_col, invalid_price=invalid_price) & (
        (pl.col(side_col) == buy_value) | (pl.col(side_col) == sell_value)
    )
    return _bounded_events_over_budget(
        lf.filter(eligible),
        x=x,
        max_points=max_points,
        polars_engine=polars_engine,
    )


def _l3_events_over_budget(
    source: Any,
    *,
    x: str,
    start: Any,
    end: Any,
    timezone: str | None,
    filters: Sequence[pl.Expr],
    price_col: str,
    size_col: str,
    side_col: str,
    action_col: str,
    bid_value: Any,
    ask_value: Any,
    actions: Mapping[str, Any],
    invalid_price: int | None,
    max_points: int,
    polars_engine: str,
) -> bool:
    lf = _windowed_l2_depth_source(
        source,
        x=x,
        start=start,
        end=end,
        timezone=timezone,
        filters=_expr_tuple(filters),
    )
    action_filter = _any_value_filter(action_col, actions.values())
    side_filter = (pl.col(side_col) == bid_value) | (pl.col(side_col) == ask_value)
    eligible = (
        _valid_price(price_col, size_col, invalid_price=invalid_price)
        & action_filter
        & side_filter
    )
    return _bounded_events_over_budget(
        lf.filter(eligible),
        x=x,
        max_points=max_points,
        polars_engine=polars_engine,
    )


def _any_value_filter(col: str, values: Any) -> pl.Expr:
    flattened = []
    for value in values:
        flattened.extend(_as_values(value))
    if not flattened:
        return pl.lit(False)
    return _value_filter(col, flattened)


def _bounded_events_over_budget(
    lf: pl.LazyFrame,
    *,
    x: str,
    max_points: int,
    polars_engine: str,
) -> bool:
    # Avoid a full count: only enough eligible rows to establish overflow are
    # allowed through the lazy query.
    found = (
        lf.select(pl.col(x))
        .limit(max_points + 1)
        .collect(engine=polars_engine)
        .height
    )
    return found > max_points


def _l2_depth_price_filter(
    source: Any,
    *,
    mode: str | None,
    l3_price_col: str,
    x: str,
    start: Any,
    end: Any,
    timezone: str | None,
    filters: Sequence[pl.Expr],
    bid_price_cols: Sequence[str],
    ask_price_cols: Sequence[str],
    bid_size_cols: Sequence[str],
    ask_size_cols: Sequence[str],
    invalid_price: int | None,
    price_scale: float | None,
    polars_engine: str,
) -> pl.Expr | None:
    if mode is None:
        return None
    if mode not in {"levels", "range"}:
        raise ValueError("l3_depth_filter must be 'levels', 'range', or None")

    lf = _windowed_l2_depth_source(
        source,
        x=x,
        start=start,
        end=end,
        timezone=timezone,
        filters=filters,
    )
    price_exprs = _depth_price_exprs(
        bid_price_cols,
        ask_price_cols,
        bid_size_cols,
        ask_size_cols,
        invalid_price=invalid_price,
        price_scale=price_scale,
    )
    if not price_exprs:
        return pl.lit(False)
    l3_price = _price_expr(
        l3_price_col,
        None,
        price_scale=price_scale,
        invalid_price=invalid_price,
    )

    if mode == "levels":
        prices = (
            lf.select(pl.concat_list(price_exprs).alias("__depth_price"))
            .explode("__depth_price")
            .drop_nulls()
            .select(pl.col("__depth_price").unique())
            .collect(engine=polars_engine)
            .get_column("__depth_price")
        )
        if prices.is_empty():
            return pl.lit(False)
        return l3_price.is_in(prices)

    bounds = (
        lf.select(
            pl.min_horizontal(price_exprs).min().alias("__low"),
            pl.max_horizontal(price_exprs).max().alias("__high"),
        )
        .collect(engine=polars_engine)
        .row(0)
    )
    low, high = bounds
    if low is None or high is None:
        return pl.lit(False)
    return l3_price.is_between(low, high)


def _windowed_l2_depth_source(
    source: Any,
    *,
    x: str,
    start: Any,
    end: Any,
    timezone: str | None,
    filters: Sequence[pl.Expr],
) -> pl.LazyFrame:
    lf = _source_to_lazy(source)
    schema = lf.collect_schema()
    x_dtype = _schema_dtype(schema, x)
    predicates = list(_expr_tuple(filters))
    start_value = _coerce_bound(start, x_dtype, timezone)
    end_value = _coerce_bound(end, x_dtype, timezone)
    if start_value is not None:
        predicates.append(pl.col(x) >= start_value)
    if end_value is not None:
        predicates.append(pl.col(x) <= end_value)
    if not predicates:
        return lf
    mask = predicates[0]
    for predicate in predicates[1:]:
        mask = mask & predicate
    return lf.filter(mask)


def _depth_price_exprs(
    bid_price_cols: Sequence[str],
    ask_price_cols: Sequence[str],
    bid_size_cols: Sequence[str],
    ask_size_cols: Sequence[str],
    *,
    invalid_price: int | None,
    price_scale: float | None,
) -> list[pl.Expr]:
    exprs = []
    for price_col, size_col in (
        *zip(bid_price_cols, bid_size_cols),
        *zip(ask_price_cols, ask_size_cols),
    ):
        exprs.append(
            _price_expr(
                price_col,
                size_col,
                price_scale=price_scale,
                invalid_price=invalid_price,
            )
        )
    return exprs


def _l3_event_layers(
    *,
    source: Any,
    x: str,
    price_col: str,
    size_col: str,
    side_col: str,
    action_col: str,
    bid_value: Any,
    ask_value: Any,
    actions: Mapping[str, Any],
    filters: Sequence[pl.Expr],
    depth_price_filter: pl.Expr | None,
    flags_col: str | None,
    label_cols: Sequence[str],
    price_scale: float | None,
    invalid_price: int | None,
    event_resample: str | None,
) -> list[dict[str, Any]]:
    layers = []
    base_filters = _expr_tuple(filters)
    valid_event = _valid_price(price_col, size_col, invalid_price=invalid_price)
    y = _price_expr(
        price_col,
        size_col,
        price_scale=price_scale,
        invalid_price=invalid_price,
    )
    for action_name, action_values in actions.items():
        action_filter = _value_filter(action_col, action_values)
        for side_name, side_value in (("bid", bid_value), ("ask", ask_value)):
            layers.append(
                {
                    "source": source,
                    "x": x,
                    "events": [
                        {
                            "y": y,
                            "name": _l3_trace_name(action_name, side_name),
                            "marker_symbol": _l3_marker_symbol(action_name, side_name),
                            "marker_color": _l3_marker_color(side_name),
                            "marker_size": _l3_marker_size(action_name),
                            "value_label": "price",
                            "label_cols": _l3_label_cols(
                                action_col,
                                side_col,
                                size_col,
                                flags_col,
                                label_cols,
                            ),
                            "resample": event_resample,
                        }
                    ],
                    "filters": _merge_filters(
                        base_filters,
                        depth_price_filter,
                        valid_event,
                        action_filter,
                        pl.col(side_col) == side_value,
                    ),
                }
            )
    return layers


def _l3_trace_name(action_name: str, side_name: str) -> str:
    side = "buy" if action_name == "trade" and side_name == "bid" else side_name
    side = "sell" if action_name == "trade" and side_name == "ask" else side
    return f"{side} {action_name}"


def _l3_marker_symbol(action_name: str, side_name: str) -> str:
    if action_name == "trade":
        return "triangle-up" if side_name == "bid" else "triangle-down"
    if action_name == "add":
        return "circle-open"
    if action_name == "modify":
        return "diamond-open"
    if action_name == "cancel":
        return "x"
    return "circle"


def _l3_marker_color(side_name: str) -> str:
    return "#00c853" if side_name == "bid" else "#ff1744"


def _l3_marker_size(action_name: str) -> int:
    return 9 if action_name == "trade" else 7


def _l3_label_cols(
    action_col: str,
    side_col: str,
    size_col: str,
    flags_col: str | None,
    label_cols: Sequence[str],
) -> tuple[str, ...]:
    cols = (action_col, side_col, size_col, flags_col, *label_cols)
    return _ordered_unique(tuple(col for col in cols if col is not None))


def _value_filter(col: str, value: Any) -> pl.Expr:
    values = _as_values(value)
    if len(values) == 1:
        return pl.col(col) == values[0]
    return pl.col(col).is_in(values)


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value]
    return list(value)


def _expr_tuple(value: Any) -> tuple[pl.Expr, ...]:
    if value is None:
        return ()
    if isinstance(value, pl.Expr):
        return (value,)
    return tuple(value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _merge_filters(*filters: pl.Expr | Sequence[pl.Expr]) -> tuple[pl.Expr, ...]:
    out = []
    for item in filters:
        out.extend(_expr_tuple(item))
    return tuple(out)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _price_expr(
    price_col: str,
    size_col: str | None,
    *,
    price_scale: float | None,
    invalid_price: int | None,
) -> pl.Expr:
    price = pl.col(price_col).cast(pl.Float64)
    if price_scale is not None:
        price = price / float(price_scale)
    return (
        pl.when(_valid_price(price_col, size_col, invalid_price=invalid_price))
        .then(price)
        .otherwise(None)
    )


def _size_expr(
    size_col: str,
    price_col: str,
    *,
    invalid_price: int | None,
) -> pl.Expr:
    return (
        pl.when(_valid_price(price_col, size_col, invalid_price=invalid_price))
        .then(pl.col(size_col))
        .otherwise(None)
    )


def _valid_price(
    price_col: str,
    size_col: str | None = None,
    *,
    invalid_price: int | None,
) -> pl.Expr:
    valid = pl.col(price_col).is_not_null() & (pl.col(price_col) > 0)
    if invalid_price is not None:
        valid = valid & (pl.col(price_col) < invalid_price)
    if size_col is not None:
        valid = valid & pl.col(size_col).is_not_null() & (pl.col(size_col) > 0)
    return valid


def _palette_color(colors: Sequence[str], idx: int) -> str:
    if not colors:
        raise ValueError("color palettes must not be empty")
    return colors[idx % len(colors)]


def _event_omission_notice(event_kind: str, event_budget: int) -> str:
    return f"{event_kind} omitted (>{event_budget:,} in selected window)"


def _price_panel_title(
    product: str | None,
    event_label: str | None,
    *,
    omitted_event_kind: str | None = None,
    event_budget: int | None = None,
) -> str:
    prefix = f"{product} " if product else ""
    base = f"{prefix}bid/ask levels"
    if omitted_event_kind is not None and event_budget is not None:
        return f"{base} — {_event_omission_notice(omitted_event_kind, event_budget)}"
    suffix = f" + {event_label}" if event_label else ""
    return f"{base}{suffix}"


def _layout(
    title: str | None,
    product: str | None,
    override: Mapping[str, Any] | None,
    *,
    event_notice: str | None = None,
    omitted_event_kind: str | None = None,
    event_budget: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "left",
            "x": 0,
        },
        "margin": {"l": 60, "r": 70, "t": 95, "b": 125},
        "hoverdistance": 1,
        "spikedistance": 1,
    }
    text = (
        title
        if title is not None
        else (f"{product} L2 order book" if product else "L2 order book")
    )
    if text:
        out["title"] = {"text": text, "x": 0.0, "xanchor": "left"}
    if override is not None:
        out.update(dict(override))
    if event_notice is not None:
        current_meta = out.get("meta")
        if isinstance(current_meta, Mapping):
            meta = dict(current_meta)
        elif current_meta is None:
            meta = {}
        else:
            meta = {"user_meta": current_meta}
        current_lob_meta = meta.get("lob_viewer")
        lob_meta = dict(current_lob_meta) if isinstance(current_lob_meta, Mapping) else {}
        lob_meta.update(
            {
                "events_omitted": True,
                "event_kind": omitted_event_kind,
                "max_event_points": event_budget,
                "notice": event_notice,
            }
        )
        meta["lob_viewer"] = lob_meta
        out["meta"] = meta
    return out


__all__ = [
    "BUY_SIDE",
    "SELL_SIDE",
    "DEFAULT_L3_ACTIONS",
    "PRICE_SCALE",
    "INVALID_PRICE",
    "plot_order_book",
]
