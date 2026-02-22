# pyright: reportReturnType=false
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

import polars as pl

from pfund_kit.style import cprint, RichColor, TextStyle
from pfolio.const import SUPPORTED_PRICE_COLUMNS
from pfolio.config import get_config
from pfolio.utils import detect_backend, to_polars, to_input_df


config = get_config()
_DEBUG_COLS = ['_trade_size']


def _prepare_data(
    df: pl.DataFrame,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> tuple[pl.DataFrame, str]:
    """Common data preparation for returns computation.

    - Validates fee/slippage basis points
    - Auto-detects ref_price: 'close' if exists, else 'price' (for tick data)
    - Adds position=1 if missing (Buy & Hold assumption)
    - Derives trade_size from position diff if missing
    - If fill_rate is provided, caps trade_size by volume participation and
      recomputes position from actual fills. 

    Returns:
        df: polars DataFrame with position and trade_size columns guaranteed
        ref_price: resolved reference price column name
    """
    # validate fee and slippage basis points
    if not 0 <= fee_bps < 10_000:
        raise ValueError(f"'fee_bps' must be between 0 and 10,000 bps, got {fee_bps}.")
    if not 0 <= slippage_bps < 10_000:
        raise ValueError(f"'slippage_bps' must be between 0 and 10,000 bps, got {slippage_bps}.")

    cols = df.columns

    ref_price = next((c for c in SUPPORTED_PRICE_COLUMNS if c in cols), None)
    assert ref_price is not None, f"No supported price column found. Expected one of {SUPPORTED_PRICE_COLUMNS}"

    has_position = 'position' in cols
    has_trade_size = 'trade_size' in cols

    # Ensure both position and trade_size columns exist
    if not has_position and not has_trade_size:
        # Buy & Hold assumption: position=1, derive trade_size from that
        df = df.with_columns(pl.lit(1).alias('position'))
        df = df.with_columns(
            pl.col('position').diff().alias('trade_size')
        ).with_columns(
            pl.when(pl.col('trade_size').is_null())
                .then(pl.col('position'))
                .otherwise(pl.col('trade_size'))
                .alias('trade_size')
        )
    elif not has_position:
        # Derive position from trade_size cumsum
        df = df.with_columns(
            pl.col('trade_size').cum_sum().alias('position')
        )
    elif not has_trade_size:
        # Derive trade_size from position diff
        df = df.with_columns(
            pl.col('position').diff().alias('trade_size')
        ).with_columns(
            # first row's diff is null → fill with position value (initial entry)
            pl.when(pl.col('trade_size').is_null())
                .then(pl.col('position'))
                .otherwise(pl.col('trade_size'))
                .alias('trade_size')
        )

    # Apply fill rate: cap trade_size by volume participation → recompute position
    if fill_rate is not None:
        if not has_position and not has_trade_size:
            cprint(
                "fill_rate ignored: no position or trade_size provided (Buy & Hold assumed)", 
                style=TextStyle.BOLD + RichColor.YELLOW
            )
        else:
            assert 0 <= fill_rate <= 1, f"fill_rate must be between 0 and 1, got {fill_rate}"
            assert 'volume' in cols, "fill_rate requires a 'volume' column in the data"
            # Preserve original ideal trade size as _trade_size for debugging
            df = df.rename({'trade_size': '_trade_size'})
            # fill_size = clip(trade_size, -max_fillable, max_fillable)
            # Null volume = unknown liquidity → treat as 0 (no fills allowed)
            max_fillable = pl.col('volume').fill_null(0) * pl.lit(fill_rate)
            trade_qty = pl.col('_trade_size').abs()
            df = df.with_columns(
                pl.when(trade_qty > max_fillable)
                    .then(
                        pl.when(pl.col('_trade_size') > 0)
                            .then(max_fillable)
                            .otherwise(pl.lit(0) - max_fillable)
                    )
                    .otherwise(pl.col('_trade_size'))
                    .alias('trade_size')
            )
            # Recompute position from cumulative fills
            df = df.with_columns(
                pl.col('trade_size').cum_sum().alias('position')
            )

    # Drop internal columns unless debug mode is on
    if not config.debug:
        to_drop = [c for c in _DEBUG_COLS if c in df.columns]
        if to_drop:
            df = df.drop(to_drop)

    return df, ref_price


def absolute_returns(
    df: IntoDataFrameT,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate absolute returns = prev_position * (p2 - p1) / p1

    Per-bar position-weighted return of the strategy.

    Args:
        df: DataFrame with 'close' or 'price' column, optionally 'position' and 'trade_size'.
        fee_bps: Trading fee in basis points, applied on trade bars.
        slippage_bps: Slippage in basis points, applied on trade bars.
        fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.
            Requires 'volume' column.

    Adds column: 'abs_rets'
    """
    _df = to_polars(df)
    _df, ref_price = _prepare_data(_df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    cost_bps = fee_bps + slippage_bps

    # raw return: prev_position * (close / prev_close - 1)
    # Row 0 has no prev price → structurally null; use when/then to set it to 0
    # so initial trade cost is preserved, while keeping nulls from missing data
    prev_position_raw = pl.col('position').shift(1)
    price_change = pl.col(ref_price) / pl.col(ref_price).shift(1) - 1
    raw_return = (
        pl.when(prev_position_raw.is_null())
            .then(pl.lit(0.0))
            .otherwise(prev_position_raw * price_change)
    )

    # trading cost: abs(trade_size) * cost_bps / 10_000 (only on trade bars)
    trade_cost = pl.col('trade_size').abs() * pl.lit(cost_bps / 10_000)

    _df = _df.with_columns(
        (raw_return - trade_cost).alias('abs_rets')
    )

    return to_input_df(_df, native_backend=detect_backend(df))


def percentage_returns(
    df: IntoDataFrameT,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate percentage returns = absolute returns * 100.

    Adds column: 'pct_rets'
    """
    _df = to_polars(
        absolute_returns(df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)
    )
    _df = _df.with_columns(
        (pl.col('abs_rets') * 100).alias('pct_rets')
    )
    return to_input_df(_df, native_backend=detect_backend(df))


def log_returns(
    df: IntoDataFrameT,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate log returns = prev_position * ln(p2 / p1)

    Note that: 
        ln(p2/p1) ≈ (p2-p1)/p1 (according to Taylor's Series)
        Approximately equal to absolute returns for small changes.

    Adds column: 'log_rets'
    """
    _df = to_polars(df)
    _df, ref_price = _prepare_data(_df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    cost_bps = fee_bps + slippage_bps

    # Row 0 has no prev price → structurally null; use when/then to set it to 0
    # so initial trade cost is preserved, while keeping nulls from missing data
    prev_position_raw = pl.col('position').shift(1)
    log_price_change = (pl.col(ref_price) / pl.col(ref_price).shift(1)).log()
    raw_return = (
        pl.when(prev_position_raw.is_null())
            .then(pl.lit(0.0))
            .otherwise(prev_position_raw * log_price_change)
    )

    trade_cost = pl.col('trade_size').abs() * pl.lit(cost_bps / 10_000)

    _df = _df.with_columns(
        (raw_return - trade_cost).alias('log_rets')
    )

    return to_input_df(_df, native_backend=detect_backend(df))


# TODO: requires benchmark data
def relative_returns(df: IntoDataFrameT, benchmark: str = '') -> IntoDataFrameT:
    """Calculate relative returns = strategy return - benchmark return."""
    raise NotImplementedError
