from typing import TypeAlias

from polars import DataFrame, Datetime, Float64, Schema

from common.baseclass import PLDF
from dataset.schema import OHLCV_PLDF

OFS_OHLCV: TypeAlias = DataFrame
OFS_OHLCV_PLDF = PLDF(
    schema=Schema(
        {
            "Date": Datetime(time_unit="us"),
            "PrevCloseOfs": Float64,  #     # 前ステップからの終値の対数差分
            "PrevVolumeOfs": Float64,  #    # 前ステップからの取引量の対数差分
            "HighOfs": Float64,  #          # 始値からの高値の対数差分
            "LowOfs": Float64,  #           # 始値からの安値の対数差分
            "CloseOfs": Float64,  #         # 始値から終値の対数差分
        }
    ),
    origins=[OHLCV_PLDF],
)
