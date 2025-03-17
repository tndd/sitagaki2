from polars import DataFrame, Datetime, Float64, Schema

from domain.common.design import Pldf
from domain.dataset.ohlcv.schema import Ohlcv


class OhlcvOfs(Pldf):
    SCHEMA = Schema(
        {
            "Date": Datetime(time_unit="us"),
            "PrevCloseOfs": Float64,  #     # 前ステップからの終値の対数差分
            "PrevVolumeOfs": Float64,  #    # 前ステップからの取引量の対数差分
            "HighOfs": Float64,  #          # 始値からの高値の対数差分
            "LowOfs": Float64,  #           # 始値からの安値の対数差分
            "CloseOfs": Float64,  #         # 始値から終値の対数差分
        }
    )
    ORIGIN = [Ohlcv]

    def __init__(self, df: DataFrame) -> None:
        super().__init__(df)
