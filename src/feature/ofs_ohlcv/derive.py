import polars as pl

from dataset.ohlcv.repository import read_df_aapl
from dataset.ohlcv.schema import OHLCV
from feature.ofs_ohlcv.schema import OFS_OHLCV, OFS_OHLCV_PLDF
from feature.ofs_ohlcv.service import (
    log_valiation_high_low_close_from_open,
    log_valiation_volume_from_open,
)


def derive_df_ofs_ohlcv(df: OHLCV) -> OFS_OHLCV:
    return (
        pl.concat(
            [
                df.select("Date"),
                log_valiation_volume_from_open(df),
                log_valiation_high_low_close_from_open(df),
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
