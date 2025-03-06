import polars as pl

from dataset.repository import read_df_aapl


def test_read_df_aapl():
    df = read_df_aapl()
    # Polars DataFrameであること
    assert isinstance(df, pl.DataFrame)
    # データが空でない
    assert len(df) > 0
    # 必要なカラムが存在しているか
    assert set(df.columns) == {"Date", "Open", "High", "Low", "Close", "Volume"}
    # Date列が日付型であること
    assert df.schema["Date"] == pl.Date
    # Dateが日付順にソートされていること"
    assert df["Date"].is_sorted()
