import polars as pl

from common.const import SCALE_BP
from dataset.ohlcv.schema import OHLCV
from feature.closes.schema import CLOSES_N4, CLOSES_N4_PLDF


def derive_closes(df: OHLCV, n: int = 4):
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

    return (
        df.select([pl.col("Date")] + lag_exprs)
        .slice(n + 1)
        .select(CLOSES_N4_PLDF.columns)
    )


def derive_closes_n4(df: OHLCV) -> CLOSES_N4:
    return derive_closes(df, 4)
