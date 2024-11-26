import polars as pl
import pandas as pd

import pfolio as po


def convert_to_polars_df(
    data: pd.DataFrame | pd.Series | pl.DataFrame | pl.Series | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(data, pl.LazyFrame):
        lf = data
        return lf if po.config.lazy else lf.collect()
    else:
        if isinstance(data, pl.DataFrame):
            df = data
        elif isinstance(data, pl.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data)
        elif isinstance(data, pd.Series):
            df = pl.from_pandas(data.to_frame())
        else:
            raise ValueError('data must be either a pandas DataFrame/Series or a polars DataFrame/Series')
        return df if not po.config.lazy else df.lazy()
