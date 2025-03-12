from pathlib import Path

import polars as pl

from dataset.ohlcv.schema import OHLCV


def read_df(name: str) -> OHLCV:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    df = (
        pl.read_csv(data_path)
        .with_columns(pl.col("Date").str.to_datetime("%m/%d/%Y"))
        .select(OHLCV.columns)
        .sort("Date")
    )
    return OHLCV(df)
