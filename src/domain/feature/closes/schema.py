from polars import DataFrame, Datetime, Float64, Schema

from domain.common.design import Pldf
from domain.dataset.ohlcv.schema import Ohlcv

"""
CLOSES:
    Closeを基準とした、直前と今の値動きの履歴を表す。
    値動きは対数比bpで表す。

    CLOSES_N{n}の"n"部分は、何日前まで終値を遡らせるかを表す。
"""


class ClosesN4(Pldf):
    # TODO: Dateは機械学習時には除外されうる
    SCHEMA = Schema(
        {
            "Date": Datetime(time_unit="us"),
            "now": Float64,  #          # 1日前(前日)を基準とした、今の終値との対数bp
            "lag_1": Float64,  #        # 2日前を基準とした、１日前の終値との対数bp
            "lag_2": Float64,  #        # 3日前を基準とした、 ~
            "lag_3": Float64,  #        # ~
            "lag_4": Float64,  #        # ~
        }
    )
    ORIGIN = [Ohlcv]

    def __init__(self, df: DataFrame) -> None:
        super().__init__(
            df,
            label="now",
        )
