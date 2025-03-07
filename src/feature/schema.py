from typing import TypeAlias

from polars import DataFrame, Date, Float64, Schema

from common.baseclass import PLDF
from dataset.schema import OHLCV_PLDF

OFS_OHLCV: TypeAlias = DataFrame
OFS_OHLCV_PLDF = PLDF(
    schema=Schema(
        {
            "Date": Date,
            "OpenOfs": Float64,  #      # 始値の昨日からの対数差分
            "HighOfs": Float64,  #      # 始値からの高値の対数差分
            "LowOfs": Float64,  #       # 始値からの安値の対数差分
            "CloseOfs": Float64,  #     # 始値から終値の対数差分
            "VolumeOfs": Float64,  #    # 取引量の昨日からの対数差分
        }
    ),
    origins=[OHLCV_PLDF],
)
