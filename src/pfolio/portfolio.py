from __future__ import annotations
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

from datetime import datetime

import polars as pl
import numpy as np
from skfolio import Portfolio as SKPortfolio
from skfolio.measures import (
    BaseMeasure,
    PerfMeasure,
    RiskMeasure,
    ExtraRiskMeasure,
    RatioMeasure,
)

from pfolio.utils import to_polars


__all__ = ['Portfolio', 'analyze']



SECONDS_PER_YEAR = 365.25 * 24 * 3600  # ~31_557_600

def _infer_bars_per_year(df: pl.DataFrame) -> float | None:
    """Infer annualized factor from timestamp data empirically.

    Counts unique bars (unique timestamps) and divides by the time span in years.
    Works for any product regardless of trading calendar (crypto 24/7, stocks market hours).

    Returns None if no timestamp column is found or if fewer than 2 unique timestamps exist.
    """
    # find the first datetime/date column
    ts_col: str | None = None
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if isinstance(dtype, (pl.Datetime, pl.Date)):
            ts_col = col_name
            break
    if ts_col is None:
        return None

    ts = df[ts_col].drop_nulls()
    n_bars = ts.n_unique()
    if n_bars < 2:
        return None

    # Use min/max so row order does not affect inference.
    first_ts = cast(datetime, ts.min())
    last_ts = cast(datetime, ts.max())
    time_span_seconds = (last_ts - first_ts).total_seconds()
    if time_span_seconds <= 0:
        return None

    time_span_years = time_span_seconds / SECONDS_PER_YEAR
    return n_bars / time_span_years


# Named metric bundles
METRIC_BUNDLES: dict[str, list[BaseMeasure]] = {
    'performance': [
        PerfMeasure.MEAN,
        PerfMeasure.ANNUALIZED_MEAN,
        RatioMeasure.SHARPE_RATIO,
        RatioMeasure.ANNUALIZED_SHARPE_RATIO,
        RatioMeasure.SORTINO_RATIO,
        RatioMeasure.CALMAR_RATIO,
    ],
    'risk': [
        RiskMeasure.STANDARD_DEVIATION,
        RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.CVAR,
        ExtraRiskMeasure.VALUE_AT_RISK,
        ExtraRiskMeasure.SKEW,
        ExtraRiskMeasure.KURTOSIS,
    ],
    'drawdown': [
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
        RatioMeasure.CALMAR_RATIO,
        RatioMeasure.AVERAGE_DRAWDOWN_RATIO,
    ],
}
METRIC_BUNDLES['full'] = (
    METRIC_BUNDLES['performance'] +
    METRIC_BUNDLES['risk'] +
    METRIC_BUNDLES['drawdown']
)


class Portfolio:
    def __init__(
        self,
        df: IntoDataFrame,
        annualized_factor: float | None = None,
        fee_bps: float = 0,
        slippage_bps: float = 0,
        fill_rate: float | None = None,
    ):
        """
        Thin wrapper that converts a DataFrame into a skfolio Portfolio.
        Args:
            df: DataFrame with a price column (prefers 'close', falls back to 'price').
                Strategy-aware mode:
                - If 'position' exists, analytics use the actual strategy exposure.
                - If 'trade_size' is missing, it is derived from position.diff().
                Fallback mode:
                - If 'position' is missing, assumes Buy & Hold (position = 1 for all bars),
                    implying one initial trade and no subsequent trades.
            annualized_factor: Annualization factor (bars per year), e.g. 252 for
                daily stock data, 365 for daily crypto, 52 for weekly.
                If None, inferred from timestamp column in the data.
                Raises ValueError if None and cannot be inferred.
            fee_bps: Trading fee in basis points, applied on trade bars.
            slippage_bps: Slippage in basis points, applied on trade bars.
            fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.
        """
        from pfolio.metrics.returns import absolute_returns

        returns_df = to_polars(absolute_returns(df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate))
        returns = returns_df['abs_rets'].drop_nulls().to_numpy()

        if annualized_factor is None:
            annualized_factor = _infer_bars_per_year(returns_df)
            if annualized_factor is None:
                raise ValueError(
                    "Cannot infer annualized_factor: no timestamp column found or insufficient data. "
                    + "Please provide annualized_factor explicitly (e.g. 252 for daily stock data, "
                    + "365 for daily crypto, 52 for weekly)."
                )

        self._skfolio = SKPortfolio(
            X=returns.reshape(-1, 1),
            weights=np.array([1.0]),
            annualized_factor=annualized_factor,
        )

    def __getattr__(self, name: str):
        """Delegate attribute access to the inner skfolio Portfolio."""
        return getattr(self._skfolio, name)


def analyze(
    df: IntoDataFrame,
    annualized_factor: float | None = None,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
    metrics: list[BaseMeasure] | None = None,
    metric_bundle: str | None = None,
) -> dict[str, float]:
    """Analyze a DataFrame and return selected metrics.
    Args:
        df: DataFrame with a price column (prefers 'close', falls back to 'price').
            Strategy-aware mode:
            - If 'position' exists, analytics use the actual strategy exposure.
            - If 'trade_size' is missing, it is derived from position.diff().
            Fallback mode:
            - If 'position' is missing, assumes Buy & Hold (position = 1 for all bars),
                implying one initial trade and no subsequent trades.
        annualized_factor: Annualization factor (bars per year), e.g. 252 for
            daily stock data, 365 for daily crypto, 52 for weekly.
            If None, inferred from timestamp column in the data.
            Raises ValueError if None and cannot be inferred.
        fee_bps: Trading fee in basis points, applied on trade bars.
        slippage_bps: Slippage in basis points, applied on trade bars.
        fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.
        metrics: List of skfolio measure enums to compute.
        metric_bundle: Metric bundle name ('performance', 'risk', 'drawdown', 'full').
            Ignored if metrics is provided.
    """
    portfolio = Portfolio(df, annualized_factor=annualized_factor, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    if metrics is None:
        selected_bundle = metric_bundle or 'full'
        if selected_bundle not in METRIC_BUNDLES:
            raise ValueError(
                f"Unknown metric_bundle '{selected_bundle}'. Supported: {list(METRIC_BUNDLES.keys())}"
            )
        metrics = METRIC_BUNDLES[selected_bundle]

    results: dict[str, float] = {}
    for measure in list(dict.fromkeys(metrics)):
        attr_name = cast(str, measure.value)  # e.g. 'sharpe_ratio', 'max_drawdown'
        value = getattr(portfolio, attr_name)
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric '{attr_name}' resolved to non-numeric value: {type(value)!r}")
        results[attr_name] = float(value)
    return results
