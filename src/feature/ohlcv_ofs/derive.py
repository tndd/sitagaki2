import polars as pl

from dataset.ohlcv.repository import read_df_aapl
from dataset.ohlcv.schema import Ohlcv
from feature.ohlcv_ofs.schema import OhlcvOfs
from feature.ohlcv_ofs.service import (
    log_valiation_high_low_close_from_open,
    log_valiation_volume_from_open,
)


def derive_df_ohlcv_ofs(ohlcv: Ohlcv) -> OhlcvOfs:
    return OhlcvOfs(
        (
            pl.concat(
                [
                    ohlcv.df.select("Date"),
                    log_valiation_volume_from_open(ohlcv.df),
                    log_valiation_high_low_close_from_open(ohlcv.df),
                ],
                how="horizontal",
            )
            .slice(1)  # 結合して1行目を削除（shiftでNaNになるため）
            .select(OhlcvOfs.get_col_names())
        )
    )


if __name__ == "__main__":
    df = read_df_aapl()
    df_ofs = derive_df_ohlcv_ofs(df)
    print(df_ofs)
