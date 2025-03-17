from polars import DataFrame, col

from domain.dataset.ohlcv.schema import Ohlcv
from domain.feature.common.const import SCALE_BP


def log_valiation_volume_from_open(df: Ohlcv) -> DataFrame:
    """
    前日を基準としたCloseとVolumeの対数差分。
    対数比の値にはbasis point単位を使用。

    DF:
        PrevCloseOfs    f64     前日Openからの対数差分
        PrevVolumeOfs   f64     前日Volumeからの対数差分
    """
    return df.select(
        [
            (col("Close") / col("Close").shift(1)).log().alias("PrevCloseOfs")
            * SCALE_BP,
            (col("Volume") / col("Volume").shift(1)).log().alias("PrevVolumeOfs")
            * SCALE_BP,
        ]
    )


def log_valiation_high_low_close_from_open(df: Ohlcv) -> DataFrame:
    """
    当日Openを基準としたHigh,Low,Closeの対数差分。
    対数比の値にはbasis point単位を使用。

    DF:
        HighOfs     f64     当日OpenからHighへの対数差分
        LowOfs      f64     当日OpenからLowへの対数差分
        CloseOfs    f64     当日OpenからのCloseへの対数差分
    """
    col_names_hlc = Ohlcv.get_col_names_hlc()
    return df.with_columns(
        [
            (col(col_name) / col("Open")).log().alias(f"{col_name}Ofs") * SCALE_BP
            for col_name in col_names_hlc
        ]
    ).select([f"{col}Ofs" for col in col_names_hlc])
