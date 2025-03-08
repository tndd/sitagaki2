import polars as pl

from common.const import SCALE_BP
from dataset.ohlcv.repository import read_df_aapl
from dataset.ohlcv.schema import OHLCV
from feature.ofs_ohlcv.schema import OFS_OHLCV, OFS_OHLCV_PLDF

COLS_HLC = ("High", "Low", "Close")


def _log_valiation_volume_from_open(df: OHLCV):
    """
    前日を基準としたCloseとVolumeの対数差分。
    対数比の値にはbasis point単位を使用。

    DF:
        PrevCloseOfs    f64     前日Openからの対数差分
        PrevVolumeOfs   f64     前日Volumeからの対数差分
    """
    return df.select(
        [
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("PrevCloseOfs")
            * SCALE_BP,
            (pl.col("Volume") / pl.col("Volume").shift(1)).log().alias("PrevVolumeOfs")
            * SCALE_BP,
        ]
    )


def _log_valiation_high_low_close_from_open(df: OHLCV):
    """
    当日Openを基準としたHigh,Low,Closeの対数差分。
    対数比の値にはbasis point単位を使用。

    DF:
        HighOfs     f64     当日OpenからHighへの対数差分
        LowOfs      f64     当日OpenからLowへの対数差分
        CloseOfs    f64     当日OpenからのCloseへの対数差分
    """
    return df.with_columns(
        [
            (pl.col(col) / pl.col("Open")).log().alias(f"{col}Ofs") * SCALE_BP
            for col in COLS_HLC
        ]
    ).select([f"{col}Ofs" for col in COLS_HLC])


def derive_df_ofs_ohlcv(df: OHLCV) -> OFS_OHLCV:
    return (
        pl.concat(
            [
                df.select("Date"),
                _log_valiation_volume_from_open(df),
                _log_valiation_high_low_close_from_open(df),
            ],
            how="horizontal",
        )
        .slice(1)  # 結合して1行目を削除（shiftでNaNになるため）
        .select(OFS_OHLCV_PLDF.columns)
    )


if __name__ == "__main__":
    df = read_df_aapl()
    df_ofs = derive_df_ofs_ohlcv(df)
    print(df_ofs)
