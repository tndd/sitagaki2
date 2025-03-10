from pathlib import Path

import polars as pl

from dataset.ohlcv.schema import OHLCV, OHLCV_PLDF


def read_df(name: str) -> OHLCV:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    return (
        pl.read_csv(data_path)
        .with_columns(pl.col("Date").str.to_datetime("%m/%d/%Y"))
        .select(OHLCV_PLDF.columns)
        .sort("Date")
    )
