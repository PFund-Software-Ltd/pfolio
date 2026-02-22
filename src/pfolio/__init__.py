# pyright: reportUnusedImport=false, reportUnsupportedDunderAll=false
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pfund_plot as plot
    from pfolio.portfolio import Portfolio, analyze
    from pfolio.utils.aliases import (
        SHARPE_RATIO, SHARPE,  # noqa: F401
        ANNUALIZED_SHARPE_RATIO, ANNUALIZED_SHARPE,  # noqa: F401
        SORTINO_RATIO, SORTINO,  # noqa: F401
        ANNUALIZED_SORTINO_RATIO, ANNUALIZED_SORTINO,  # noqa: F401
        CALMAR_RATIO, CALMAR,  # noqa: F401
        MAX_DRAWDOWN, MAX_DD, MDD,  # noqa: F401
        AVERAGE_DRAWDOWN, AVG_DD,  # noqa: F401
        CVAR,  # noqa: F401
        VALUE_AT_RISK, VAR,  # noqa: F401
        MEAN,  # noqa: F401
        ANNUALIZED_MEAN,  # noqa: F401
        VARIANCE,  # noqa: F401
        STANDARD_DEVIATION, STD,  # noqa: F401
        SEMI_DEVIATION,  # noqa: F401
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
