# Portfolio Analytics & Optimization

[![Twitter Follow](https://img.shields.io/twitter/follow/pfund_ai?style=social)](https://x.com/pfund_ai)
![PyPI downloads](https://img.shields.io/pypi/dm/pfolio?label=downloads)
[![PyPI](https://img.shields.io/pypi/v/pfolio.svg)](https://pypi.org/project/pfolio)
![PyPI - Support Python Versions](https://img.shields.io/pypi/pyversions/pfolio)
[![Discussions](https://img.shields.io/badge/Discussions-Let's%20Chat-green)](https://github.com/PFund-Software-Ltd/pfolio/discussions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PFund-Software-Ltd/pfolio)
<!-- ![GitHub stars](https://img.shields.io/github/stars/PFund-Software-Ltd/pfolio?style=social) -->
<!-- [![afterpython](https://afterpython.org/shield.svg)](https://afterpython.org) -->

[pfund]: https://github.com/PFund-Software-Ltd/pfund
[pfeed]: https://github.com/PFund-Software-Ltd/pfeed
[pytrade.org]: https://pytrade.org
[skfolio]: https://github.com/skfolio/skfolio
[polars]: https://github.com/pola-rs/polars

## Problem
Most portfolio analytics libraries are shallow — they expect a simple price series and assume **Buy and Hold (position = 1 throughout)**. This makes them useless for evaluating real trading strategies, where positions change, trades have costs, and fills aren't always complete.

## Solution
`pfolio` computes metrics from **realistic backtest dataframes** with strategy-aware columns (`position`, `trade_size`, `avg_price`, etc.), factoring in fees, slippage, and fill rate.

---
`pfolio` is a portfolio analytics and optimization library built on [skfolio], using [polars] to compute accurate metrics from **realistic backtest DataFrames** — not just buy-and-hold price series.

> `pfolio` is part of the [PFund] ecosystem but works as a **standalone package** — any DataFrame in the right shape works

## Core Features
- [x] Accurate backtest metrics — Sharpe, Sortino, Calmar, MDD, CVaR, VaR, and more
- [x] Trading-specific stats — win rate, payoff ratio, win/loss streaks, holding periods
- [ ] Multi-asset portfolio backtest analytics
- [ ] Portfolio optimization

---

<details>
<summary>Table of Contents</summary>

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Related Projects](#related-projects)
- [Disclaimer](#disclaimer)

</details>

## Installation

```bash
pip install pfolio
```


## Quick Start

`backtest_df` is any DataFrame that includes strategy-aware columns alongside standard OHLCV data. Key columns pfolio uses (subset shown):

| date | close | position | trade_price | trade_size | avg_price |
|------|-------|----------|-------------|------------|-----------|
| 2024-01-01 | 96500 | 0 | null | 0.0 | null |
| 2024-01-02 | 97200 | 1 | 97200 | 1.0 | 97200 |
| 2024-01-03 | 98100 | 1 | null | 0.0 | 97200 |
| 2024-01-04 | 97800 | 0 | 97800 | -1.0 | null |

```python
import pfolio as po

# One-shot analysis — returns a dict of metrics
results = po.analyze(backtest_df, annualized_factor=252)

# Use a preset bundle: 'performance', 'risk', 'drawdown', 'trading', or 'full' (default)
results = po.analyze(backtest_df, annualized_factor=252, metric_bundle='performance')

# Pick specific metrics
results = po.analyze(backtest_df, annualized_factor=252, metrics=[po.SHARPE, po.MDD, po.SORTINO])

# Factor in transaction costs
results = po.analyze(backtest_df, annualized_factor=252, fee_bps=10, slippage_bps=5)
```

For multiple queries on the same data, use the `Portfolio` object directly:

```python
portfolio = po.Portfolio(backtest_df, annualized_factor=252)

portfolio.annualized_sharpe_ratio
portfolio.max_drawdown
portfolio.win_rate
portfolio.holding_periods   # array of bars held per trade
portfolio.win_streaks       # array of consecutive winning trades
```

> `annualized_factor` is inferred automatically from the timestamp column if not provided (e.g. 252 for daily stocks, 365 for daily crypto).

All of `skfolio`'s measures are supported — see its [documentation](https://skfolio.org/) for the full list.

---
## Related Projects
- [pfund] — A Complete Algo-Trading Framework for Machine Learning, TradFi, CeFi and DeFi ready. Supports Vectorized and Event-Driven Backtesting, Paper and Live Trading
- [pfeed] — Data engine for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.
- [pytrade.org] - A curated list of Python libraries and resources for algorithmic trading.



## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This framework is intended for educational and research purposes only. It should not be used for real trading without understanding the risks involved. Trading in financial markets involves significant risk, and there is always the potential for loss. Your trading results may vary. No representation is being made that any account will or is likely to achieve profits or losses similar to those discussed on this platform.

The developers of this framework are not responsible for any financial losses incurred from using this software. This includes but not limited to losses resulting from inaccuracies in any financial data output by PFeed. Users should conduct their due diligence, verify the accuracy of any data produced by PFeed, and consult with a professional financial advisor before engaging in real trading activities.
