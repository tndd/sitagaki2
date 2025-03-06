from typing import TypeAlias

import polars as pl

OHCLV: TypeAlias = pl.DataFrame
OHCLV_SCHEMA = pl.Schema(
    {
        "Date": pl.Date,
        "Open": pl.Float64,
        "High": pl.Float64,
        "Low": pl.Float64,
        "Close": pl.Float64,
        "Volume": pl.Int64,
    }
)
