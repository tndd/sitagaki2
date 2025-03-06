from dataset.repository import read_df_aapl
from dataset.schema import OHCLV, OHCLV_PLDF


def test_read_df_aapl():
    df = read_df_aapl()
    # OHCLVであること(Polars dataframe)
    assert isinstance(df, OHCLV)
    # データが空でない
    assert len(df) > 0
    # スキーマの完全一致チェック
    assert df.schema == OHCLV_PLDF.schema
    # Dateが日付順にソートされていること"
    assert df["Date"].is_sorted()
