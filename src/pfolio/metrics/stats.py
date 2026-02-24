from enum import Enum

import numpy as np
from numpy.typing import NDArray


class CustomMeasure(Enum):
    """Custom measures not provided by skfolio."""
    WIN_RATE = 'win_rate'
    AVG_WIN = 'avg_win'
    AVG_LOSS = 'avg_loss'
    PAYOFF_RATIO = 'payoff_ratio'
    PROFIT_FACTOR = 'profit_factor'
    TIME_IN_MARKET = 'time_in_market'


def win_rate(pnl: NDArray[np.floating], exclude_flat: bool = True) -> float:
    """Fraction of profitable events.

    Args:
        pnl: 1-D array of P&L values. Pass dollar_pnls for bar-level win rate,
            or rpnl filtered to non-zero rows for trade-level win rate.
        exclude_flat: If True (default), exclude zero-P&L bars from the
            denominator — they are neither wins nor losses. If False, zero-P&L
            bars count as non-wins, lowering the rate.

    Returns:
        Win rate as a float in [0.0, 1.0], or nan if no valid bars exist.
    """
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if exclude_flat:
        pnl = pnl[pnl != 0]
    if len(pnl) == 0:
        return np.nan
    return float((pnl > 0).sum() / len(pnl))


def avg_win(pnl: NDArray[np.floating]) -> float:
    """Mean P&L of winning events only.

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        Average winning P&L (positive), or nan if no wins exist.
    """
    pnl = np.asarray(pnl, dtype=float)
    wins = pnl[pnl > 0]
    if len(wins) == 0:
        return np.nan
    return float(wins.mean())


def avg_loss(pnl: NDArray[np.floating]) -> float:
    """Mean P&L of losing events only. Returns a negative number.

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        Average losing P&L (negative), or nan if no losses exist.
    """
    pnl = np.asarray(pnl, dtype=float)
    losses = pnl[pnl < 0]
    if len(losses) == 0:
        return np.nan
    return float(losses.mean())


def payoff_ratio(pnl: NDArray[np.floating]) -> float:
    """Average win magnitude divided by average loss magnitude.

    payoff_ratio = avg_win / |avg_loss|

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        Payoff ratio (> 1 means wins are larger than losses on average),
        or nan if no wins or no losses exist.
    """
    win = avg_win(pnl)
    loss = avg_loss(pnl)
    if np.isnan(win) or np.isnan(loss):
        return np.nan
    return float(win / abs(loss))


def profit_factor(pnl: NDArray[np.floating]) -> float:
    """Total gross profit divided by total gross loss.

    profit_factor = Σ pnl_i (pnl_i > 0) / |Σ pnl_j (pnl_j < 0)|

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        Profit factor (> 1 means profitable overall), inf if no losses,
        or nan if no wins and no losses.
    """
    pnl = np.asarray(pnl, dtype=float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return np.nan if gross_profit == 0 else np.inf
    return float(gross_profit / gross_loss)


def _consecutive_runs(mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Return lengths of consecutive True runs in a boolean array.

    Example: [T, T, F, T, T, T, F] → [2, 3]

    Args:
        mask: 1-D boolean array.

    Returns:
        1-D array of run lengths (empty if no True values).
    """
    if len(mask) == 0 or not mask.any():
        return np.array([], dtype=int)
    # Pad with False on both ends so diff catches start/end of runs
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts


def win_streaks(pnl: NDArray[np.floating]) -> NDArray[np.intp]:
    """Lengths of each consecutive winning streak.

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        1-D array of streak lengths (empty if no wins).
    """
    pnl = np.asarray(pnl, dtype=float)
    return _consecutive_runs(pnl > 0)


def loss_streaks(pnl: NDArray[np.floating]) -> NDArray[np.intp]:
    """Lengths of each consecutive losing streak.

    Args:
        pnl: 1-D array of P&L values.

    Returns:
        1-D array of streak lengths (empty if no losses).
    """
    pnl = np.asarray(pnl, dtype=float)
    return _consecutive_runs(pnl < 0)


def time_in_market(position: NDArray[np.floating]) -> float:
    """Fraction of bars where the strategy holds a position.

    time_in_market = count(position ≠ 0) / N

    Args:
        position: 1-D array of position values.

    Returns:
        Ratio in [0.0, 1.0], or nan if array is empty.
    """
    position = np.asarray(position, dtype=float)
    if len(position) == 0:
        return np.nan
    return float((position != 0).sum() / len(position))


def holding_periods(position: NDArray[np.floating]) -> NDArray[np.intp]:
    """Durations (in bars) of each continuous position.

    A holding period starts when position transitions from 0 to non-zero
    (or flips sign), and ends when it returns to 0 or flips sign again.

    Args:
        position: 1-D array of position values.

    Returns:
        1-D array of holding period lengths (empty if never in market).
    """
    import polars as pl
    position = pl.Series(position).forward_fill().fill_null(0).to_numpy()
    if len(position) == 0:
        return np.array([], dtype=int)
    signs = np.sign(position)
    # Label each bar with a group ID that increments on every sign change.
    # Consecutive bars with the same sign share a group.
    changes = np.concatenate(([True], signs[1:] != signs[:-1]))
    group_ids = np.cumsum(changes)
    # Keep only in-market bars (sign != 0), then count per group
    in_market = signs != 0
    if not in_market.any():
        return np.array([], dtype=int)
    _, counts = np.unique(group_ids[in_market], return_counts=True)
    return counts


def drawdown_periods(drawdowns: NDArray[np.floating]) -> NDArray[np.intp]:
    """Durations (in bars) of each drawdown episode.

    A drawdown episode starts when the drawdown goes below 0 (equity drops
    below its peak) and ends when it recovers to 0 (new peak). If the series
    ends underwater, the final unrecovered episode is included.

    Args:
        drawdowns: 1-D array of drawdown values (≤ 0, where 0 = at peak).
            Use skfolio's portfolio.drawdowns or compute as
            cumulative_returns - cumulative_max(cumulative_returns).

    Returns:
        1-D array of drawdown episode lengths (empty if no drawdowns).
    """
    drawdowns = np.asarray(drawdowns, dtype=float)
    return _consecutive_runs(drawdowns < 0)
