from dataset.ohlcv.repository import read_df_aapl
from dataset.ohlcv.schema import Ohlcv


def test_read_df_aapl():
    ohlcv = read_df_aapl()
    # OHLCVであること(Polars dataframe)
    assert isinstance(ohlcv, Ohlcv)
    # データが空でない
    assert len(ohlcv.df) > 0
    # スキーマの完全一致チェック
    assert ohlcv.schema == Ohlcv.schema
    # Dateが日付順にソートされていること"
    assert ohlcv.df["Date"].is_sorted()
