# pyright: reportUnusedImport=false, reportUnsupportedDunderAll=false
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pfund_plot as plot
    from pfolio.portfolio import Portfolio, analyze
    from pfolio.utils.aliases import (
        # PerfMeasure
        MEAN as MEAN,
        ANNUALIZED_MEAN as ANNUALIZED_MEAN,
        # RiskMeasure
        VARIANCE as VARIANCE,
        ANNUALIZED_VARIANCE as ANNUALIZED_VARIANCE,
        SEMI_VARIANCE as SEMI_VARIANCE,
        ANNUALIZED_SEMI_VARIANCE as ANNUALIZED_SEMI_VARIANCE,
        STANDARD_DEVIATION as STANDARD_DEVIATION,
        STD as STD,
        ANNUALIZED_STANDARD_DEVIATION as ANNUALIZED_STANDARD_DEVIATION,
        ANNUALIZED_STD as ANNUALIZED_STD,
        SEMI_DEVIATION as SEMI_DEVIATION,
        ANNUALIZED_SEMI_DEVIATION as ANNUALIZED_SEMI_DEVIATION,
        MEAN_ABSOLUTE_DEVIATION as MEAN_ABSOLUTE_DEVIATION,
        MAD as MAD,
        CVAR as CVAR,
        EVAR as EVAR,
        WORST_REALIZATION as WORST_REALIZATION,
        CDAR as CDAR,
        MAX_DRAWDOWN as MAX_DRAWDOWN,
        MAX_DD as MAX_DD,
        MDD as MDD,
        AVERAGE_DRAWDOWN as AVERAGE_DRAWDOWN,
        AVG_DD as AVG_DD,
        EDAR as EDAR,
        FIRST_LOWER_PARTIAL_MOMENT as FIRST_LOWER_PARTIAL_MOMENT,
        FLPM as FLPM,
        ULCER_INDEX as ULCER_INDEX,
        GINI_MEAN_DIFFERENCE as GINI_MEAN_DIFFERENCE,
        # ExtraRiskMeasure
        VALUE_AT_RISK as VALUE_AT_RISK,
        VAR as VAR,
        DRAWDOWN_AT_RISK as DRAWDOWN_AT_RISK,
        DAR as DAR,
        ENTROPIC_RISK_MEASURE as ENTROPIC_RISK_MEASURE,
        FOURTH_CENTRAL_MOMENT as FOURTH_CENTRAL_MOMENT,
        FOURTH_LOWER_PARTIAL_MOMENT as FOURTH_LOWER_PARTIAL_MOMENT,
        SKEW as SKEW,
        KURTOSIS as KURTOSIS,
        # RatioMeasure
        SHARPE_RATIO as SHARPE_RATIO,
        SHARPE as SHARPE,
        ANNUALIZED_SHARPE_RATIO as ANNUALIZED_SHARPE_RATIO,
        ANNUALIZED_SHARPE as ANNUALIZED_SHARPE,
        SORTINO_RATIO as SORTINO_RATIO,
        SORTINO as SORTINO,
        ANNUALIZED_SORTINO_RATIO as ANNUALIZED_SORTINO_RATIO,
        ANNUALIZED_SORTINO as ANNUALIZED_SORTINO,
        MEAN_ABSOLUTE_DEVIATION_RATIO as MEAN_ABSOLUTE_DEVIATION_RATIO,
        MAD_RATIO as MAD_RATIO,
        FIRST_LOWER_PARTIAL_MOMENT_RATIO as FIRST_LOWER_PARTIAL_MOMENT_RATIO,
        FLPM_RATIO as FLPM_RATIO,
        VALUE_AT_RISK_RATIO as VALUE_AT_RISK_RATIO,
        VAR_RATIO as VAR_RATIO,
        CVAR_RATIO as CVAR_RATIO,
        ENTROPIC_RISK_MEASURE_RATIO as ENTROPIC_RISK_MEASURE_RATIO,
        EVAR_RATIO as EVAR_RATIO,
        WORST_REALIZATION_RATIO as WORST_REALIZATION_RATIO,
        DRAWDOWN_AT_RISK_RATIO as DRAWDOWN_AT_RISK_RATIO,
        DAR_RATIO as DAR_RATIO,
        CDAR_RATIO as CDAR_RATIO,
        CALMAR_RATIO as CALMAR_RATIO,
        CALMAR as CALMAR,
        AVERAGE_DRAWDOWN_RATIO as AVERAGE_DRAWDOWN_RATIO,
        AVG_DD_RATIO as AVG_DD_RATIO,
        EDAR_RATIO as EDAR_RATIO,
        ULCER_INDEX_RATIO as ULCER_INDEX_RATIO,
        GINI_MEAN_DIFFERENCE_RATIO as GINI_MEAN_DIFFERENCE_RATIO,
    )

from importlib.metadata import version
from pfolio.config import get_config, configure


__version__ = version('pfolio')


# All alias names from utils/aliases.py
_ALIASES = (
    # PerfMeasure
    'MEAN',
    'ANNUALIZED_MEAN',
    # RiskMeasure
    'VARIANCE', 'ANNUALIZED_VARIANCE',
    'SEMI_VARIANCE', 'ANNUALIZED_SEMI_VARIANCE',
    'STANDARD_DEVIATION', 'STD',
    'ANNUALIZED_STANDARD_DEVIATION', 'ANNUALIZED_STD',
    'SEMI_DEVIATION', 'ANNUALIZED_SEMI_DEVIATION',
    'MEAN_ABSOLUTE_DEVIATION', 'MAD',
    'CVAR', 'EVAR',
    'WORST_REALIZATION',
    'CDAR',
    'MAX_DRAWDOWN', 'MAX_DD', 'MDD',
    'AVERAGE_DRAWDOWN', 'AVG_DD',
    'EDAR',
    'FIRST_LOWER_PARTIAL_MOMENT', 'FLPM',
    'ULCER_INDEX',
    'GINI_MEAN_DIFFERENCE',
    # ExtraRiskMeasure
    'VALUE_AT_RISK', 'VAR',
    'DRAWDOWN_AT_RISK', 'DAR',
    'ENTROPIC_RISK_MEASURE',
    'FOURTH_CENTRAL_MOMENT',
    'FOURTH_LOWER_PARTIAL_MOMENT',
    'SKEW',
    'KURTOSIS',
    # RatioMeasure
    'SHARPE_RATIO', 'SHARPE',
    'ANNUALIZED_SHARPE_RATIO', 'ANNUALIZED_SHARPE',
    'SORTINO_RATIO', 'SORTINO',
    'ANNUALIZED_SORTINO_RATIO', 'ANNUALIZED_SORTINO',
    'MEAN_ABSOLUTE_DEVIATION_RATIO', 'MAD_RATIO',
    'FIRST_LOWER_PARTIAL_MOMENT_RATIO', 'FLPM_RATIO',
    'VALUE_AT_RISK_RATIO', 'VAR_RATIO',
    'CVAR_RATIO',
    'ENTROPIC_RISK_MEASURE_RATIO',
    'EVAR_RATIO',
    'WORST_REALIZATION_RATIO',
    'DRAWDOWN_AT_RISK_RATIO', 'DAR_RATIO',
    'CDAR_RATIO',
    'CALMAR_RATIO', 'CALMAR',
    'AVERAGE_DRAWDOWN_RATIO', 'AVG_DD_RATIO',
    'EDAR_RATIO',
    'ULCER_INDEX_RATIO',
    'GINI_MEAN_DIFFERENCE_RATIO',
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
