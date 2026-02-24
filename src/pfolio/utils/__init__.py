# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from dask.dataframe import DataFrame as DaskDataFrame
    from narwhals.typing import IntoDataFrame
    from pyspark.sql import DataFrame as SparkDataFrame

import types

import narwhals as nw
import polars as pl


def detect_backend(df: IntoDataFrame) -> types.ModuleType:
    """Detect the native backend module of a DataFrame (e.g. pandas, polars)."""
    return nw.get_native_namespace(nw.from_native(df, eager_only=True))


def to_polars(df: IntoDataFrame) -> pl.DataFrame:
    """Convert any DataFrame to polars DataFrame.

    Uses narwhals as the conversion engine to support pandas, polars,
    and any other DataFrame type narwhals understands.
    If the input is already a polars DataFrame, returns it directly.
    """
    if isinstance(df, pl.DataFrame):
        return df
    return nw.from_native(df, eager_only=True).to_polars()


def to_input_df(
    df: pl.DataFrame, native_backend: types.ModuleType
) -> pd.DataFrame | pl.DataFrame | DaskDataFrame | SparkDataFrame:
    """Convert polars DataFrame back to the caller's native backend."""
    backend_name = native_backend.__name__
    if backend_name == "polars":
        return df
    elif backend_name == "pandas":
        return df.to_pandas()
    elif "dask" in backend_name:
        import dask.dataframe as dd

        return dd.from_pandas(df.to_pandas(), npartitions=1)
    elif "pyspark" in backend_name:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(df.to_pandas())
    else:
        raise ValueError(f"Unsupported native backend: {backend_name}")
