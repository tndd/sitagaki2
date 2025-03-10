import polars as pl

from common.const import SCALE_BP
from dataset.ohlcv.schema import OHLCV
from feature.closes.schema import CLOSES_N4, CLOSES_N4_PLDF


def derive_closes_n4(df: OHLCV) -> CLOSES_N4:
    return (
        df.select(
            [
                pl.col("Date"),
                (pl.col("Close") / pl.col("Close").shift(1)).log().alias("now")
                * SCALE_BP,
                (pl.col("Close").shift(1) / pl.col("Close").shift(2))
                .log()
                .alias("lag_1")
                * SCALE_BP,
                (pl.col("Close").shift(2) / pl.col("Close").shift(3))
                .log()
                .alias("lag_2")
                * SCALE_BP,
                (pl.col("Close").shift(3) / pl.col("Close").shift(4))
                .log()
                .alias("lag_3")
                * SCALE_BP,
                (pl.col("Close").shift(4) / pl.col("Close").shift(5))
                .log()
                .alias("lag_4")
                * SCALE_BP,
            ]
        )
        .slice(5)
        .select(CLOSES_N4_PLDF.columns)
    )
