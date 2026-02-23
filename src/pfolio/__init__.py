# pyright: reportUnusedImport=false, reportUnsupportedDunderAll=false
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pfund_plot as plot
    from pfolio.portfolio import Portfolio, analyze
    from pfolio.utils.aliases import (
        SHARPE_RATIO as SHARPE_RATIO,
        SHARPE as SHARPE,
        ANNUALIZED_SHARPE_RATIO as ANNUALIZED_SHARPE_RATIO,
        ANNUALIZED_SHARPE as ANNUALIZED_SHARPE,
        SORTINO_RATIO as SORTINO_RATIO,
        SORTINO as SORTINO,
        ANNUALIZED_SORTINO_RATIO as ANNUALIZED_SORTINO_RATIO,
        ANNUALIZED_SORTINO as ANNUALIZED_SORTINO,
        CALMAR_RATIO as CALMAR_RATIO,
        CALMAR as CALMAR,
        MAX_DRAWDOWN as MAX_DRAWDOWN,
        MAX_DD as MAX_DD,
        MDD as MDD,
        AVERAGE_DRAWDOWN as AVERAGE_DRAWDOWN,
        AVG_DD as AVG_DD,
        CVAR as CVAR,
        VALUE_AT_RISK as VALUE_AT_RISK,
        VAR as VAR,
        MEAN as MEAN,
        ANNUALIZED_MEAN as ANNUALIZED_MEAN,
        VARIANCE as VARIANCE,
        STANDARD_DEVIATION as STANDARD_DEVIATION,
        STD as STD,
        SEMI_DEVIATION as SEMI_DEVIATION,
    )

from importlib.metadata import version
from pfolio.config import get_config, configure


__version__ = version('pfolio')


# All alias names from utils/aliases.py
_ALIASES = (
    'SHARPE_RATIO', 'SHARPE',
    'ANNUALIZED_SHARPE_RATIO', 'ANNUALIZED_SHARPE',
    'SORTINO_RATIO', 'SORTINO',
    'ANNUALIZED_SORTINO_RATIO', 'ANNUALIZED_SORTINO',
    'CALMAR_RATIO', 'CALMAR',
    'MAX_DRAWDOWN', 'MAX_DD', 'MDD',
    'AVERAGE_DRAWDOWN', 'AVG_DD',
    'CVAR',
    'VALUE_AT_RISK', 'VAR',
    'MEAN',
    'ANNUALIZED_MEAN',
    'VARIANCE',
    'STANDARD_DEVIATION', 'STD',
    'SEMI_DEVIATION',
)


def __getattr__(name: str):
    if name == 'plot':
        import pfund_plot as plot
        return plot
    elif name == 'Portfolio':
        from pfolio.portfolio import Portfolio
        return Portfolio
    elif name == 'analyze':
        from pfolio.portfolio import analyze
        return analyze
    elif name in _ALIASES:
        from pfolio.utils import aliases
        return getattr(aliases, name)
    raise AttributeError(f"module 'pfolio' has no attribute '{name}'")


__all__ = (
    '__version__',
    'get_config',
    'configure',
    'plot',
    'Portfolio',
    'analyze',
    *_ALIASES,
)
def __dir__():
    return sorted(__all__)
