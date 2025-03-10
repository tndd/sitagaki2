from dataset.ohlcv.service import read_df


def read_df_aapl():
    return read_df("AAPL")


def read_df_amd():
    return read_df("AMD")


def read_df_sbux():
    return read_df("SBUX")


if __name__ == "__main__":
    # データを日付順にソート
    df = read_df_aapl()
    print(df)
