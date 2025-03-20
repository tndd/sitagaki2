from polars import DataFrame, Datetime, Float64, Schema, col

from domain.common.design import Pldf
from domain.dataset.ohlcv.schema import Ohlcv
from domain.feature.common.const import SCALE_BP

"""
CLOSES:
    Closeを基準とした、直前と今の値動きの履歴を表す。
    値動きは対数比bpで表す。

    CLOSES_N{n}の"n"部分は、何日前まで終値を遡らせるかを表す。
"""


class ClosesN4(Pldf):
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
            exclude="Date",
        )


def derive_closes_n4(ohlcv: Ohlcv) -> ClosesN4:
    return ClosesN4(_derive_closes(ohlcv, 4).select(ClosesN4.get_col_names()))


def _derive_closes(ohlcv: Ohlcv, n: int) -> DataFrame:
    # 基準となるシフト量を動的に生成
    base_shifts = [col("Close").shift(i) for i in range(n + 2)]
    # 各lagカラムの計算式を生成
    lag_exprs = [
        (base_shifts[i] / base_shifts[i + 1])
        .log()
        .alias(f"lag_{i}" if i > 0 else "now")
        * SCALE_BP
        for i in range(n + 1)
    ]
    return ohlcv.df.select([col("Date")] + lag_exprs).slice(n + 1)
