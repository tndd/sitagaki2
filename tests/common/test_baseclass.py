from polars import DataFrame, Date, Float64, Int64, Schema

from common.baseclass import PLDF


def test_PLDF():
    # クラス変数の定義
    PLDF.schema = Schema(
        {
            "A_DT": Date,
            "B_FL": Float64,
            "C_IN": Int64,
        }
    )
    PLDF.origins = None
    # インスタンス変数dfを宣言しつつ実体化
    pldf = PLDF(df=DataFrame())
    assert isinstance(pldf, PLDF)
    # カラム名が定義の通りの並びになってるか？
    assert pldf.get_col_names() == ["A_DT", "B_FL", "C_IN"]
