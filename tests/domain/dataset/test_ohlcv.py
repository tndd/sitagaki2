from domain.dataset.ohlcv import Ohlcv, read_df_aapl


def test_read_df_aapl():
    ohlcv = read_df_aapl()
    # OHLCVであること(Polars dataframe)
    assert isinstance(ohlcv, Ohlcv)
    # データが空でない
    assert len(ohlcv.df) > 0
    # スキーマの完全一致チェック
    assert ohlcv.df.schema == Ohlcv.SCHEMA
    # Dateが日付順にソートされていること"
    assert ohlcv.df["Date"].is_sorted()
