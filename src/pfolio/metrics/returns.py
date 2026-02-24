# pyright: reportReturnType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

import polars as pl

from pfolio.metrics.pnls import dollar_pnls, get_contract_expressions
from pfolio.utils import detect_backend, to_input_df, to_polars


def absolute_returns(
    df: IntoDataFrameT,
    initial_capital: float = 1_000_000,
    contract_type: Literal["linear", "inverse"] = "linear",
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
    normalize_by: Literal["equity", "notional"] = "notional",
) -> IntoDataFrameT:
    """Calculate absolute returns per bar.

    Args:
        df: DataFrame with 'close' or 'price' column, optionally 'position' and 'trade_size'.
        initial_capital: Starting capital in dollars. Default 1,000,000.
            Only used when normalize_by='equity' (to build the equity curve).
            Ignored when normalize_by='notional'.
        contract_type: 'linear' (default) or 'inverse'.
        contract_multiplier: For linear: dollar value per price point per contract.
            For inverse: USD notional per contract.
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

    Adds columns: 'pnl', 'ret' (and 'equity' when normalize_by='equity')
    """
    if normalize_by == "equity" and initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")

    _df = to_polars(
        dollar_pnls(
            df,
            contract_type=contract_type,
            contract_multiplier=contract_multiplier,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            fill_rate=fill_rate,
        )
    )

    if normalize_by == "equity":
        # equity = initial_capital + cumsum(pnl)
        _df = _df.with_columns(
            (pl.lit(initial_capital) + pl.col("pnl").cum_sum()).alias("equity")
        )
        # ret = pnl / prev_equity
        # For bar 0, prev_equity = initial_capital (equity before any P&L)
        prev_equity = (
            pl.col("equity").shift(1).forward_fill().fill_null(pl.lit(initial_capital))
        )
        _df = _df.with_columns((pl.col("pnl") / prev_equity).alias("ret"))
    elif normalize_by == "notional":
        # ret = pnl / notional_exposure
        # Notional is contract-type-aware (from get_contract_expressions).
        from pfolio.const import SUPPORTED_PRICE_COLUMNS

        ref_price = next(c for c in SUPPORTED_PRICE_COLUMNS if c in _df.columns)
        cost_rate = (fee_bps + slippage_bps) / 10_000
        prev_position = pl.col("position").shift(1)
        price = pl.col(ref_price)
        _raw_pnl, _trade_cost, notional = get_contract_expressions(
            contract_type,
            cost_rate,
            contract_multiplier,
            exposure=prev_position,
            current_price=price,
            base_price=pl.col(ref_price).shift(1),
            trade_qty=pl.col("trade_size").abs(),
            cost_price=price,
        )
        _df = _df.with_columns(
            pl.when((notional == 0) | notional.is_null())
            .then(pl.lit(0.0))
            .otherwise(pl.col("pnl") / notional)
            .alias("ret")
        )
    else:
        raise ValueError(
            f"normalize_by must be 'equity' or 'notional', got '{normalize_by}'"
        )

    return to_input_df(_df, native_backend=detect_backend(df))


def log_returns(
    df: IntoDataFrameT,
    initial_capital: float = 1_000_000,
    contract_type: Literal["linear", "inverse"] = "linear",
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
    normalize_by: Literal["equity", "notional"] = "notional",
) -> IntoDataFrameT:
    """Calculate log returns = ln(1 + ret).

    Note that:
        ln(p2/p1) ≈ (p2-p1)/p1 (according to Taylor's Series)
        Approximately equal to absolute returns for small changes.

    Adds columns: 'pnl', 'ret', 'log_ret' (and 'equity' when normalize_by='equity')
    """
    _df = to_polars(
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
    _df = _df.with_columns((1 + pl.col("ret")).log().alias("log_ret"))
    return to_input_df(_df, native_backend=detect_backend(df))


# TODO: requires benchmark data
def relative_returns(df: IntoDataFrameT, benchmark: str = "") -> IntoDataFrameT:
    """Calculate relative returns = strategy return - benchmark return."""
    raise NotImplementedError
