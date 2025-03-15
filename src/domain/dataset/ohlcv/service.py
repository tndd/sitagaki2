from pathlib import Path

import polars as pl

from domain.dataset.ohlcv.schema import Ohlcv


def read_df(name: str) -> Ohlcv:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    df = (
        pl.read_csv(data_path)
        .with_columns(pl.col("Date").str.to_datetime("%m/%d/%Y"))
        .select(Ohlcv.get_col_names())
        .sort("Date")
    )
    return Ohlcv(df)
