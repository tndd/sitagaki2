from pathlib import Path

import polars as pl

from dataset.schema import OHLCV, OHLCV_PLDF


def _read_df(name: str) -> OHLCV:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    return (
        pl.read_csv(data_path)
        .with_columns(pl.col("Date").str.to_date("%m/%d/%Y"))
        .select(OHLCV_PLDF.columns)
        .sort("Date")
    )


def read_df_aapl():
    return _read_df("AAPL")


def read_df_amd():
    return _read_df("AMD")


def read_df_sbux():
    return _read_df("SBUX")


if __name__ == "__main__":
    # データを日付順にソート
    df = read_df_aapl()
    print(df)
