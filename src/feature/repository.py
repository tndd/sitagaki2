import polars as pl

from dataset.repository import read_df_aapl
from dataset.schema import OHCLV
from feature.schema import OFS_OHCLV, OFS_OHCLV_PLDF

COLS_HLC = ("High", "Low", "Close")


def _log_valiation_volume_from_open(df: OHCLV):
    """
    前日を基準としたOpenとVolumeの対数差分。
    対数比の値にはbasis point単位を使用。

    DF:
        OpenOfs     f64     前日Openからの対数差分
        VolumeOfs   f64     前日Volumeからの対数差分
    """
    return df.select(
        [
            (pl.col("Open") / pl.col("Open").shift(1)).log().alias("OpenOfs") * 10000,
            (pl.col("Volume") / pl.col("Volume").shift(1)).log().alias("VolumeOfs")
            * 10000,
        ]
    )


def _log_valiation_high_low_close_from_open(df: OHCLV):
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
            (pl.col(col) / pl.col("Open")).log().alias(f"{col}Ofs") * 10000
            for col in COLS_HLC
        ]
    ).select([f"{col}Ofs" for col in COLS_HLC])


def derive_ofs_ohclv(df: OHCLV) -> OFS_OHCLV:
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
        .select(OFS_OHCLV_PLDF.columns)
    )


if __name__ == "__main__":
    df = read_df_aapl()
    df_ofs = derive_ofs_ohclv(df)
    print(df_ofs)
