from typing import TypeAlias

from polars import DataFrame, Date, Float64, Int64, Schema

from common.baseclass import PLDF

OHCLV: TypeAlias = DataFrame
OHCLV_PLDF = PLDF(
    schema=Schema(
        {
            "Date": Date,
            "Open": Float64,
            "High": Float64,
            "Low": Float64,
            "Close": Float64,
            "Volume": Int64,
        }
    ),
    origins=None,
)
