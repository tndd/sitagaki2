from typing import TypeAlias

from polars import DataFrame, Datetime, Float64, Int64, Schema

from common.baseclass import PLDF

OHLCV: TypeAlias = DataFrame
OHLCV_PLDF = PLDF(
    schema=Schema(
        {
            "Date": Datetime(time_unit="us"),
            "Open": Float64,
            "High": Float64,
            "Low": Float64,
            "Close": Float64,
            "Volume": Int64,
        }
    ),
    origins=None,
)
