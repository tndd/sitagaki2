from os import path

import pandas as pd

# 現在のスクリプトファイルのディレクトリパス
SELF_PATH = path.dirname(path.abspath(__file__))


def _read_df(name: str):
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し
    """
    data_path = path.join(SELF_PATH, "data", f"{name}.csv")
    return pd.read_csv(
        data_path,
        parse_dates=["Date"],
        date_format="%m/%d/%Y",
        index_col="Date",
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
