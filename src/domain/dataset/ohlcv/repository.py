from domain.dataset.ohlcv.schema import Ohlcv
from domain.dataset.ohlcv.service import read_df


def read_df_aapl() -> Ohlcv:
    return read_df("AAPL")


def read_df_amd() -> Ohlcv:
    return read_df("AMD")


def read_df_sbux() -> Ohlcv:
    return read_df("SBUX")


if __name__ == "__main__":
    # データを日付順にソート
    df = read_df_aapl()
    print(df)
