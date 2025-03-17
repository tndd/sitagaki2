from polars import DataFrame, Date, Float64, Int64, Schema

from domain.common.design import Pldf


def test_PLDF():
    # クラス変数の定義
    Pldf.SCHEMA = Schema(
        {
            "A_DT": Date,
            "B_FL": Float64,
            "C_IN": Int64,
        }
    )
    Pldf.ORIGIN = None
    # インスタンス変数dfを宣言しつつ実体化
    pldf = Pldf(df=DataFrame())
    assert isinstance(pldf, Pldf)
    # カラム名が定義の通りの並びになってるか？
    assert pldf.get_col_names() == ["A_DT", "B_FL", "C_IN"]
