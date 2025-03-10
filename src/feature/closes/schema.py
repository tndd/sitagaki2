from typing import TypeAlias

from polars import DataFrame, Datetime, Float64, Schema

from common.baseclass import PLDF
from dataset.ohlcv.schema import OHLCV_PLDF

"""
CLOSES:
    Closeを基準とした、直前と今の値動きの履歴を表す。
    値動きは対数比bpで表す。

    CLOSES_N{n}の"n"部分は、データセットの値動きを表すカラムの数を表す。
"""

CLOSES_N5: TypeAlias = DataFrame
CLOSES_N5_PLDF = PLDF(
    schema=Schema(
        {
            "Date": Datetime(time_unit="us"),
            "now": Float64,  #          # 1日前(前日)を基準とした、今の終値との対数bp
            "lag_1": Float64,  #        # 2日前を基準とした、１日前の終値との対数bp
            "lag_2": Float64,  #        # 3日前を基準とした、 ~
            "lag_3": Float64,  #        # ~
            "lag_4": Float64,  #        # ~
        }
    ),
    origins=[OHLCV_PLDF],
)
