import polars as pl

from common.const import SCALE_BP
from dataset.ohlcv.schema import Ohlcv


def derive_closes(ohlcv: Ohlcv, n: int):
    # 基準となるシフト量を動的に生成
    base_shifts = [pl.col("Close").shift(i) for i in range(n + 2)]

    # 各lagカラムの計算式を生成
    lag_exprs = [
        (base_shifts[i] / base_shifts[i + 1])
        .log()
        .alias(f"lag_{i}" if i > 0 else "now")
        * SCALE_BP
        for i in range(n + 1)
    ]

    return ohlcv.df.select([pl.col("Date")] + lag_exprs).slice(n + 1)
