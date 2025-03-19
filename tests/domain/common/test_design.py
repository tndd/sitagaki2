from numpy import ndarray
from pandas import DataFrame as DataFramePd
from polars import DataFrame, Date, Float64, Int64, Schema

from domain.common.design import LabeledDataset, Pldf
from fixture.factory.feature.closes import factory_closes_n4


def test_pldf():
    """
    pldfの簡易テスト
    """
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


def test_pldf_practical():
    """
    より実践的なpldfの詳細テスト
    """
    # まずテスト対象がPldfであるかを確認
    closes_pldf = factory_closes_n4()
    assert isinstance(closes_pldf, Pldf)
    # LabeledDatasetの取得
    labeled_ds = closes_pldf.get_labeled_dataset()
    assert isinstance(labeled_ds, LabeledDataset)
    assert isinstance(labeled_ds.X, DataFramePd)
    assert isinstance(labeled_ds.y, ndarray)
