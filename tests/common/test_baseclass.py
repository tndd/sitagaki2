from polars import Date, Float64, Int64, Schema

from common.baseclass import PLDF


def test_PLDF():
    pldf = PLDF(
        schema=Schema(
            {
                "A_DT": Date,
                "B_FL": Float64,
                "C_IN": Int64,
            }
        ),
        origins=None,
    )
    # インスタンスの作成が成功してるか
    assert isinstance(pldf, PLDF)
    # カラム名が定義の通りの並びになってるか？
    assert pldf.columns == ["A_DT", "B_FL", "C_IN"]
