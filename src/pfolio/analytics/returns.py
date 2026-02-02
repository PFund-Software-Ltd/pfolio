import numpy as np
import polars as pl
import pandas as pd

import pfolio as po
from pfolio.utils import utils


def absolute_returns(
    data: pl.Series | pd.Series | pl.DataFrame | pd.DataFrame | pl.LazyFrame,
    separate_returns=False,
    # TODO:
    fee=0,
    fee_type='bps',
    slippage=0,
    slippage_type='bps',
    debug: bool=False,
) -> pl.DataFrame | pl.LazyFrame:
    '''Calculate absolute returns = p2 - p1
    Args:
        ref_price: reference price column name, if 'close' column is not found and 
            ref_price is not specified, 'price' column will be used (e.g. for tick data)
        separate_returns: if True, return realized_pnl and unrealized_pnl columns
    '''
    df: pl.DataFrame | pl.LazyFrame = utils.convert_to_polars_df(data)
    cols = df.columns
    # use 'close' column by default, if not exists, use 'price' column (e.g. for tick data)
    ref_price = 'close' if 'close' in cols else 'price'
    assert ref_price in cols, f"Reference price column '{ref_price}' not found in the data"
    no_position = 'position' not in cols
    no_trade_size = 'trade_size' not in cols
    # TODO: what about trade_price?
    # add position column with 1s if not exists = assume Buy & Hold strategy
    if no_position:
        df = df.with_columns(pl.lit(1).alias('position'))
    if no_trade_size:
        df = (
            df.with_columns(
                pl.col('position')
                    .diff()
                    .alias('trade_size')
            # fill the first trade size with the position value
            ).with_columns(
                pl.when(pl.arange(pl.len()) == 0)
                    .then(pl.col('position'))
                    .otherwise(pl.col('trade_size'))
                    .alias('trade_size')
            )
        )
    df = (
        # 1. determine if the trades are for offsetting position or not
        df.with_columns(
            pl.when(
                    (pl.col('trade_size') != 0) & 
                    (pl.col('position').shift(1) != 0) &
                    (pl.col('trade_size').sign() != pl.col('position').shift(1).sign())
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias('_offset'),
        # 2. calculate rolling entry and exit prices
        ).with_columns(
            pl.when(pl.col('_offset'))
                .then(pl.col('trade_price'))
                .otherwise(
                    pl.when(pl.col('position') != 0)
                        .then(pl.col(ref_price))
                        .otherwise(pl.lit(None))
                )
                .alias('_rolling_exit_price'),
            pl.when(pl.col('_offset'))
                .then(pl.col(ref_price).shift(1))
                .otherwise(
                    pl.when(pl.col('position') != 0)
                        .then(
                            ( pl.col(ref_price).shift(1).fill_null(0) * pl.col('position').shift(1).fill_null(0)
                            + pl.col('trade_price') * pl.col('trade_size') ) 
                            / pl.col('position')
                        )
                        .otherwise(pl.lit(None))
                )
                .alias('_rolling_entry_price'),
        # 3. calculate realized and unrealized pnls
        ).with_columns(
            pl.when(~pl.col('_offset'))
                .then( (pl.col('_rolling_exit_price') - pl.col('_rolling_entry_price') ) * pl.col('position').sign() )
                .otherwise(pl.lit(None))
                .alias('upnl'),
            pl.when(pl.col('_offset'))
                .then( (pl.col('_rolling_exit_price') - pl.col('_rolling_entry_price') ) * (-pl.col('trade_size').sign()) )
                .otherwise(pl.lit(None))
                .alias('rpnl')
        # 4. calculate absolute returns
        ).with_columns(
            pl.coalesce(['rpnl', 'upnl'])
            .alias('abs_rets')
        )
    )
    
    if not separate_returns:
        df = df.select(pl.all().exclude('rpnl', 'upnl'))

    if not debug and not po.config.debug:
        df = df.select(pl.all().exclude(
            [col for col in df.columns if col.startswith('_')]
        ))

    # drop position column if it was not in the original data
    if no_position:
        df = df.drop('position')
    if no_trade_size:
        df = df.drop('trade_size')
    return df
        

def relative_returns(
    data: pl.Series | pd.Series | pl.DataFrame | pd.DataFrame | pl.LazyFrame,
    trade_price: str='trade_price',
    trade_size: str='trade_size',
    ref_price: str='close',
    separate_returns=False,
    debug: bool=False,
):
    '''Calculate relative returns = (p2-p1)/p1'''
    # df = absolute_returns(
    #     data, 
    #     'trade_price'='trade_price', 
    #     'trade_size'='trade_size', 
    #     ref_price=ref_price, 
    #     debug=True,
    #     separate_returns=True)
    # df = df.with_columns(
    #     (pl.col('abs_rets') / pl.col('_rolling_entry_price'))
    #     .alias('rel_rets')
    # )
    # if not debug and not po.config.debug:
    #     df = df.select(pl.all().exclude(
    #         [col for col in df.columns if col.startswith('_')]
    #     ))
    # return df
    pass


def percentage_returns(data):
    pass


# TODO
def comparative_returns(data, benchmark=''):
    pass


# TODO
def log_returns(data):
    '''Calculate log returns = ln(p2/p1) ≈ (p2-p1)/p1 (according to Taylor's Series)'''
    pass
