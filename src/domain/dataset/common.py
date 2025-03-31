from pathlib import Path

import pandas as pd
from pandas import DataFrame, read_csv


def read_df(name: str) -> DataFrame:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    df = read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").set_index("Date")
    return df


if __name__ == "__main__":
    df = read_df("aapl")
    print(df)
