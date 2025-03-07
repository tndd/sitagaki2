from dataset.repository import read_df_aapl
from dataset.schema import OHLCV, OHLCV_PLDF


def test_read_df_aapl():
    df = read_df_aapl()
    # OHLCVであること(Polars dataframe)
    assert isinstance(df, OHLCV)
    # データが空でない
    assert len(df) > 0
    # スキーマの完全一致チェック
    assert df.schema == OHLCV_PLDF.schema
    # Dateが日付順にソートされていること"
    assert df["Date"].is_sorted()
