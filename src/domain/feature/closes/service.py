from polars import DataFrame, col

from domain.dataset.ohlcv.schema import Ohlcv
from domain.feature.common.const import SCALE_BP


def calc_features_closes(df: DataFrame, n: int) -> DataFrame:
    """
    外部から汎用的に利用可能にするために、
    pl.Dataframeを直接受け取る機能を分離した。
    """
    base_shifts = [col("Close").shift(i) for i in range(n + 2)]
    lag_exprs = [
        (base_shifts[i] / base_shifts[i + 1])
        .log()
        .alias(f"lag_{i}" if i > 0 else "now")
        * SCALE_BP
        for i in range(n + 1)
    ]
    return df.select([col("Date")] + lag_exprs).slice(n + 1)


def derive_closes(ohlcv: Ohlcv, n: int) -> DataFrame:
    return calc_features_closes(ohlcv.df, n)
