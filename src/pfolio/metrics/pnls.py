# pyright: reportReturnType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

import polars as pl

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
    """Common data preparation for P&L computation.

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

    null_count = df[ref_price].null_count()
    if null_count > 0:
        raise ValueError(
            f"'{ref_price}' contains {null_count} null value(s). " +
            "Clean your data before passing it to pfolio."
        )

    has_position = 'position' in cols
    has_trade_size = 'trade_size' in cols

    # Critical invariant before any arithmetic: no nulls in state columns.
    # - trade_size nulls are treated as "no trade" and set to 0.
    #   Otherwise trade_cost becomes null, and dollar_pnls = raw_pnl - trade_cost
    #   also becomes null.
    # - position must be null-free before math. We forward-fill to preserve
    #   exposure continuity, then fill any leading nulls with 0 so
    #   position.diff() and downstream P&L calculations behave deterministically.
    # NOTE: Keeping these columns null-free prevents silent null propagation through P&L,
    # equity, and return calculations.
    if has_trade_size:
        df = df.with_columns(pl.col('trade_size').fill_null(0))
    if has_position:
        df = df.with_columns(pl.col('position').forward_fill().fill_null(0))

    
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
                .alias('trade_size').fill_null(0)
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
                .alias('trade_size').fill_null(0)
        )


    # Apply fill rate: cap trade_size by volume participation → recompute position
    if fill_rate is not None:
        if not has_position and not has_trade_size:
            print("WARNING: fill_rate ignored: no position or trade_size provided (Buy & Hold assumed)")
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


def get_contract_expressions(
    contract_type: Literal['linear', 'inverse'],
    cost_rate: float,
    contract_multiplier: float,
    *,
    exposure: pl.Expr,
    current_price: pl.Expr,
    base_price: pl.Expr,
    trade_qty: pl.Expr,
    cost_price: pl.Expr,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Build raw_pnl, trade_cost, and notional expressions for the given contract type.

    All positional/price inputs are parameterized so callers can reuse this for
    dollar_pnls (prev_position, price, prev_price), unrealized_pnls (position,
    price, avg_price), and realized_pnls (offset_qty, price, avg_price).

    Args:
        exposure: Signed quantity bearing the price movement
            (e.g. prev_position, position, offset_qty * sign(prev_position)).
        current_price: Mark-to-market price.
        base_price: Reference price (e.g. prev_price, avg_price).
        trade_qty: Absolute quantity for cost calculation.
        cost_price: Price at which cost is computed. For linear contracts, cost
            is proportional to this price. For inverse contracts, this is ignored
            (cost is price-independent).

    Returns:
        (raw_pnl, trade_cost, notional) polars expressions.
    """
    multiplier = pl.lit(contract_multiplier)
    diff = current_price - base_price

    if contract_type == 'linear':
        raw_pnl = exposure * diff * multiplier
        trade_cost = trade_qty * cost_price * pl.lit(cost_rate) * multiplier
        notional = exposure.abs() * base_price * multiplier
    elif contract_type == 'inverse':
        raw_pnl = exposure * multiplier * diff / base_price
        trade_cost = trade_qty * multiplier * pl.lit(cost_rate)
        notional = exposure.abs() * multiplier
    else:
        raise ValueError(f"Unknown contract_type '{contract_type}'. Supported: 'linear', 'inverse'.")

    # Null guard: if exposure is null (e.g. first bar with shift), raw_pnl = 0
    raw_pnl = (
        pl.when(exposure.is_null())
            .then(pl.lit(0.0))
            .otherwise(raw_pnl)
    )

    return raw_pnl, trade_cost, notional


def dollar_pnls(
    df: IntoDataFrameT,
    contract_type: Literal['linear', 'inverse'] = 'linear',
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate dollar P&L per bar (mark-to-market, conceptually daily-settled).

    Linear contracts:
        raw_pnl = prev_position * (price - prev_price) * multiplier
        trade_cost = abs(trade_size) * price * cost_rate * multiplier

    Inverse contracts (fixed USD notional per contract):
        raw_pnl = prev_position * multiplier * (price - prev_price) / prev_price
        trade_cost = abs(trade_size) * multiplier * cost_rate

    Args:
        df: DataFrame with 'close' or 'price' column, optionally 'position' and 'trade_size'.
        contract_type: 'linear' (default) or 'inverse'.
        contract_multiplier: For linear: dollar value per price point per contract (e.g. ES=50).
            For inverse: USD notional per contract (e.g. 100 if 1 contract = 100 USD).
        fee_bps: Trading fee in basis points, applied on trade bars.
        slippage_bps: Slippage in basis points, applied on trade bars.
        fill_rate: Volume participation rate (0-1). Caps fill size at fill_rate * volume.

    Adds column: 'pnl'
    """
    _df = to_polars(df)
    _df, ref_price = _prepare_data(_df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    cost_rate = (fee_bps + slippage_bps) / 10_000
    prev_position = pl.col('position').shift(1)
    price = pl.col(ref_price)

    raw_pnl, trade_cost, _notional = get_contract_expressions(
        contract_type, cost_rate, contract_multiplier,
        exposure=prev_position,
        current_price=price,
        base_price=pl.col(ref_price).shift(1),
        trade_qty=pl.col('trade_size').abs(),
        cost_price=price,
    )

    _df = _df.with_columns(
        (raw_pnl - trade_cost).alias('pnl')
    )

    return to_input_df(_df, native_backend=detect_backend(df))


def unrealized_pnls(
    df: IntoDataFrameT,
    contract_type: Literal['linear', 'inverse'] = 'linear',
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate unrealized P&L per bar (mark-to-market vs avg_price, net of entry costs).

    Requires an 'avg_price' column in the DataFrame (average entry price).
    Entry costs are subtracted using the contract-type-aware trade_cost formula
    (proportional to avg_price for linear, flat for inverse).

    Adds column: 'upnl'
    """
    _df = to_polars(df)
    assert 'avg_price' in _df.columns, "unrealized_pnls requires an 'avg_price' column"
    _df, ref_price = _prepare_data(_df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    cost_rate = (fee_bps + slippage_bps) / 10_000
    position = pl.col('position')
    price = pl.col(ref_price)
    avg_price = pl.col('avg_price')

    raw_upnl, entry_cost, _ = get_contract_expressions(
        contract_type, cost_rate, contract_multiplier,
        exposure=position,
        current_price=price,
        base_price=avg_price,
        trade_qty=position.abs(),
        cost_price=avg_price,
    )

    _df = _df.with_columns((raw_upnl - entry_cost).alias('upnl'))

    return to_input_df(_df, native_backend=detect_backend(df))


def realized_pnls(
    df: IntoDataFrameT,
    contract_type: Literal['linear', 'inverse'] = 'linear',
    contract_multiplier: float = 1,
    fee_bps: float = 0,
    slippage_bps: float = 0,
    fill_rate: float | None = None,
) -> IntoDataFrameT:
    """Calculate per-bar realized P&L (locked in when position reduces).

    Requires an 'avg_price' column in the DataFrame (average entry price).
    Both entry costs (at avg_price) and exit costs
    (at current price) are deducted using contract-type-aware formulas.

    For position flips (e.g. long→short), only the closing portion is realized;
    the opening portion starts a new position.

    Adds column: 'rpnl' (per-bar realized)
    """
    _df = to_polars(df)
    assert 'avg_price' in _df.columns, "realized_pnls requires an 'avg_price' column"
    _df, ref_price = _prepare_data(_df, fee_bps=fee_bps, slippage_bps=slippage_bps, fill_rate=fill_rate)

    cost_rate = (fee_bps + slippage_bps) / 10_000
    price = pl.col(ref_price)
    # Use prev bar's avg_price: on flip bars, current avg_price reflects the
    # NEW position's entry price, but we need the OLD position's cost basis.
    avg_price = pl.col('avg_price').shift(1)
    prev_position = pl.col('position').shift(1).fill_null(0)
    trade_size = pl.col('trade_size')

    # A trade is closing when it opposes the previous position
    is_offset = (trade_size.sign() != prev_position.sign()) & (prev_position != 0) & (trade_size != 0)

    # offset_qty: the portion of trade_size that closes the previous position
    # For a flip (e.g. prev=10, trade=-15), offset_qty = min(15, 10) = 10
    offset_qty = pl.min_horizontal(trade_size.abs(), prev_position.abs())
    offset_size = offset_qty * prev_position.sign()

    # Raw realized PnL + entry cost (at avg_price)
    raw_rpnl, entry_cost, _ = get_contract_expressions(
        contract_type, cost_rate, contract_multiplier,
        exposure=offset_size,
        current_price=price,
        base_price=avg_price,
        trade_qty=offset_qty,
        cost_price=avg_price,
    )

    # Exit cost (at current price)
    _, exit_cost, _ = get_contract_expressions(
        contract_type, cost_rate, contract_multiplier,
        exposure=offset_size,
        current_price=price,
        base_price=avg_price,
        trade_qty=offset_qty,
        cost_price=price,
    )

    _df = _df.with_columns(
        pl.when(is_offset)
            .then(raw_rpnl - entry_cost - exit_cost)
            .otherwise(pl.lit(0.0))
            .alias('rpnl')
    )

    return to_input_df(_df, native_backend=detect_backend(df))
