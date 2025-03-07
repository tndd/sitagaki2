import polars as pl

from feature.repository import derive_df_ofs_ohclv


def test_derive_df_ofs_ohclv():
    # テスト用のOHCLVに則ったデータフレームを作成
    df = pl.DataFrame()
    df_ofs = derive_df_ofs_ohclv(df)
