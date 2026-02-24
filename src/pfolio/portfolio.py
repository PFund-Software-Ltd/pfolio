# ruff: noqa: I001
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

from datetime import datetime
from functools import cached_property

import numpy as np
import polars as pl
from numpy.typing import NDArray
from skfolio import Portfolio as SKPortfolio
from skfolio.measures import (
    BaseMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)

from pfolio.metrics.stats import (
    CustomMeasure,
    win_rate as _win_rate,
    avg_win as _avg_win,
    avg_loss as _avg_loss,
    payoff_ratio as _payoff_ratio,
    profit_factor as _profit_factor,
    time_in_market as _time_in_market,
    win_streaks as _win_streaks,
    loss_streaks as _loss_streaks,
    holding_periods as _holding_periods,
    drawdown_periods as _drawdown_periods,
)
from pfolio.utils import to_polars

__all__ = ["Portfolio", "analyze"]


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
METRIC_BUNDLES: dict[str, list[BaseMeasure | CustomMeasure]] = {
    "performance": [
        PerfMeasure.MEAN,
        PerfMeasure.ANNUALIZED_MEAN,
        RatioMeasure.SHARPE_RATIO,
        RatioMeasure.ANNUALIZED_SHARPE_RATIO,
        RatioMeasure.SORTINO_RATIO,
        RatioMeasure.ANNUALIZED_SORTINO_RATIO,
        RatioMeasure.CALMAR_RATIO,
    ],
    "risk": [
        RiskMeasure.STANDARD_DEVIATION,
        RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
        RiskMeasure.CVAR,
        RiskMeasure.CDAR,
        ExtraRiskMeasure.VALUE_AT_RISK,
        ExtraRiskMeasure.SKEW,
        ExtraRiskMeasure.KURTOSIS,
    ],
    "drawdown": [
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        ExtraRiskMeasure.DRAWDOWN_AT_RISK,
        RatioMeasure.DRAWDOWN_AT_RISK_RATIO,
        RatioMeasure.AVERAGE_DRAWDOWN_RATIO,
    ],
    "trading": [
        CustomMeasure.WIN_RATE,
        CustomMeasure.AVG_WIN,
        CustomMeasure.AVG_LOSS,
        CustomMeasure.PAYOFF_RATIO,
        CustomMeasure.PROFIT_FACTOR,
        CustomMeasure.TIME_IN_MARKET,
    ],
}
METRIC_BUNDLES["full"] = (
    METRIC_BUNDLES["performance"]
    + METRIC_BUNDLES["risk"]
    + METRIC_BUNDLES["drawdown"]
    + METRIC_BUNDLES["trading"]
)


class Portfolio:
    def __init__(
        self,
        df: IntoDataFrame,
        initial_capital: float = 1_000_000,
        annualized_factor: float | None = None,
        contract_type: Literal["linear", "inverse"] = "linear",
        contract_multiplier: float = 1,
        fee_bps: float = 0,
        slippage_bps: float = 0,
        fill_rate: float | None = None,
        normalize_by: Literal["equity", "notional"] = "notional",
        risk_free_rate: float = 0,
        **skfolio_kwargs: Any,
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
            initial_capital: Starting capital in dollars. Default 1,000,000.
                Only used when normalize_by='equity' (to build the equity curve).
                Ignored when normalize_by='notional'.
            annualized_factor: Annualization factor (bars per year), e.g. 252 for
                daily stock data, 365 for daily crypto, 52 for weekly.
                If None, inferred from timestamp column in the data.
                Raises ValueError if None and cannot be inferred.
            contract_type: 'linear' (default) or 'inverse'.
            contract_multiplier: For linear: dollar value per price point per contract (e.g. ES=50).
                For inverse: USD notional per contract (e.g. 100 if 1 contract = 100 USD).
            fee_bps: Trading fee in basis points, applied on trade bars.
            slippage_bps: Slippage in basis points, applied on trade bars.
            fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.
            normalize_by: How to normalize dollar P&L into returns.
                - 'notional' (default): ret = dollar_pnl / notional_exposure.
                  Measures the strategy's edge per unit of exposure, ignoring
                  idle cash. Use for pfund's vectorized/hybrid backtests where
                  order size is fixed and cannot adapt to equity changes.
                - 'equity': ret = dollar_pnl / prev_equity.
                  Measures return on total capital including idle cash.
                  Use for pfund's event-driven backtests where the strategy
                  dynamically sizes orders based on current capital.
            risk_free_rate: Annual risk-free rate (as a decimal, e.g. 0.04 for 4%).
                Used by excess-return ratios like Sharpe and Sortino. Default 0.
            **skfolio_kwargs: Additional keyword arguments forwarded to skfolio's Portfolio
                (e.g. cvar_beta, value_at_risk_beta, min_acceptable_return).
                See skfolio.Portfolio documentation for all options.
        """
        from pfolio.metrics.returns import absolute_returns

        self._df = to_polars(
            absolute_returns(
                df,
                initial_capital=initial_capital,
                contract_type=contract_type,
                contract_multiplier=contract_multiplier,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                fill_rate=fill_rate,
                normalize_by=normalize_by,
            )
        )
        returns = self._df["ret"].drop_nulls().to_numpy()

        if annualized_factor is None:
            annualized_factor = _infer_bars_per_year(self._df)
            if annualized_factor is None:
                raise ValueError(
                    "Cannot infer annualized_factor: no timestamp column found or insufficient data. "
                    + "Please provide annualized_factor explicitly (e.g. 252 for daily stock data, "
                    + "365 for daily crypto, 52 for weekly)."
                )

        # compounded=False (cumsum) because notional-normalized returns don't
        # track a reinvested capital base — position size is fixed each bar,
        # so cumprod would overstate cumulative returns by assuming reinvestment
        # that never happened (and often can't due to lot sizes, min quantities, etc.).
        # equity-normalized returns DO track a real capital base, so cumprod is correct.
        compounded = normalize_by == "equity"
        self._skfolio = SKPortfolio(
            X=returns.reshape(-1, 1),
            weights=np.array([1.0]),
            annualized_factor=annualized_factor,
            compounded=compounded,
            risk_free_rate=risk_free_rate,
            **skfolio_kwargs,
        )

    @cached_property
    def _dollar_pnls(self) -> NDArray[np.floating]:
        return self._df["pnl"].drop_nulls().to_numpy()

    @cached_property
    def _position(self) -> NDArray[np.floating]:
        return self._df["position"].to_numpy()

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    def total_pnl(self) -> float:
        """Total P&L in dollars: sum(dollar_pnls)."""
        return float(self._dollar_pnls.sum())

    # Custom stats: maps attr name → (function, data attribute)
    _CUSTOM_STATS: ClassVar[dict[str, tuple[Any, str]]] = {
        "win_rate": (_win_rate, "_dollar_pnls"),
        "avg_win": (_avg_win, "_dollar_pnls"),
        "avg_loss": (_avg_loss, "_dollar_pnls"),
        "payoff_ratio": (_payoff_ratio, "_dollar_pnls"),
        "profit_factor": (_profit_factor, "_dollar_pnls"),
        "time_in_market": (_time_in_market, "_position"),
        "win_streaks": (_win_streaks, "_dollar_pnls"),
        "loss_streaks": (_loss_streaks, "_dollar_pnls"),
        "holding_periods": (_holding_periods, "_position"),
    }

    def __getattr__(self, name: str):
        """Delegate attribute access to custom stats or skfolio Portfolio."""
        if name in self._CUSTOM_STATS:
            func, data_attr = self._CUSTOM_STATS[name]
            return func(object.__getattribute__(self, data_attr))
        if name == "drawdown_periods":
            return _drawdown_periods(
                cast(NDArray[np.floating], self._skfolio.drawdowns)
            )
        return getattr(self._skfolio, name)


def analyze(
    df: IntoDataFrame,
    initial_capital: float = 1_000_000,
    annualized_factor: float | None = None,
    contract_type: Literal["linear", "inverse"] = "linear",
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
    normalize_by: Literal["equity", "notional"] = "notional",
    risk_free_rate: float = 0,
    metrics: list[BaseMeasure | CustomMeasure] | None = None,
    metric_bundle: str | None = None,
    **skfolio_kwargs: Any,
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
        initial_capital: Starting capital in dollars. Default 1,000,000.
            Only used when normalize_by='equity'. Ignored when normalize_by='notional'.
        annualized_factor: Annualization factor (bars per year), e.g. 252 for
            daily stock data, 365 for daily crypto, 52 for weekly.
            If None, inferred from timestamp column in the data.
            Raises ValueError if None and cannot be inferred.
        contract_type: 'linear' (default) or 'inverse'.
        contract_multiplier: For linear: dollar value per price point per contract (e.g. ES=50).
            For inverse: USD notional per contract (e.g. 100 if 1 contract = 100 USD).
        fee_bps: Trading fee in basis points, applied on trade bars.
        slippage_bps: Slippage in basis points, applied on trade bars.
        fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.
        normalize_by: How to normalize dollar P&L into returns. Default 'notional'.
            See Portfolio docstring for details.
        risk_free_rate: Annual risk-free rate (as a decimal, e.g. 0.04 for 4%).
            Used by excess-return ratios like Sharpe and Sortino. Default 0.
        metrics: List of skfolio measure enums to compute.
        metric_bundle: Metric bundle name ('performance', 'risk', 'drawdown', 'full').
            Ignored if metrics is provided.
        **skfolio_kwargs: Additional keyword arguments forwarded to skfolio's Portfolio
            (e.g. cvar_beta, value_at_risk_beta, min_acceptable_return).
            See skfolio.Portfolio documentation for all options.
    """
    portfolio = Portfolio(
        df,
        initial_capital=initial_capital,
        annualized_factor=annualized_factor,
        contract_type=contract_type,
        contract_multiplier=contract_multiplier,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        fill_rate=fill_rate,
        normalize_by=normalize_by,
        risk_free_rate=risk_free_rate,
        **skfolio_kwargs,
    )

    if metrics is None:
        selected_bundle = metric_bundle or "full"
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
            raise TypeError(
                f"Metric '{attr_name}' resolved to non-numeric value: {type(value)!r}"
            )
        results[attr_name] = float(value)
    return results
