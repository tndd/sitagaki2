import polars as pl

from dataset.repository import read_df_aapl
from dataset.schema import OHCLV
from feature.schema import OFS_OHCLV, OFS_OHCLV_PLDF

COLS_HLC = ("High", "Low", "Close")


def _log_valiation_volume_from_open(df: OHCLV):
    return df.select(
        [
            (pl.col("Open") / pl.col("Open").shift(1)).log().alias("OpenOfs") * 10000,
            (pl.col("Volume") / pl.col("Volume").shift(1)).log().alias("VolumeOfs")
            * 10000,
        ]
    )


def _log_valiation_high_low_close_from_open(df: OHCLV):
    return df.with_columns(
        [
            (pl.col(col) / pl.col("Open")).log().alias(f"{col}Ofs") * 10000
            for col in COLS_HLC
        ]
    )


def derive_ofs_ohclv(df: OHCLV) -> OFS_OHCLV:
    df_ofs_ov = _log_valiation_volume_from_open(df)
    df_ofs_hlc = _log_valiation_high_low_close_from_open(df)
    # 結合して1行目を削除（shiftでNaNになるため）
    return (
        pl.concat(
            [
                df.select("Date"),
                df_ofs_ov,
                df_ofs_hlc.select([f"{col}Ofs" for col in COLS_HLC]),
            ],
            how="horizontal",
        )
        .slice(1)
        .select(OFS_OHCLV_PLDF.columns)
    )


if __name__ == "__main__":
    df = read_df_aapl()
    df_ofs = derive_ofs_ohclv(df)
    print(df_ofs)
