from typing import TypeAlias

import polars as pl

from common.baseclass import PLDF

OHLCV: TypeAlias = pl.DataFrame
OHLCV_PLDF = PLDF(
    schema=pl.Schema(
        {
            "Date": pl.Datetime(time_unit="us"),
            "Open": pl.Float64,
            "High": pl.Float64,
            "Low": pl.Float64,
            "Close": pl.Float64,
            "Volume": pl.Int64,
        }
    ),
    origins=None,
)
