from lightgbm import Dataset
from numpy import ndarray
from pandas import DataFrame as DataFramePD
from polars import DataFrame, Date, Float64, Int64, Schema

from domain.common.design import (
    LabeledDataset,
    LabeledDatasetSplit,
    Pldf,
)
from fixture.factory.feature.closes import (
    factory_closes_n4_1000,
    factory_closes_n4_cycle,
)


### Pldf ###
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


def test_pldf_pddf():
    """
    PandasDFへの変換とインデックス設定が正常に行われているかを確認する。
    """
    pldf = factory_closes_n4_1000()
    assert isinstance(pldf.pddf, DataFramePD)
    assert pldf.pddf.index.name == "Date"


def test_pldf_get_labeled_dataset():
    # まずテスト対象がPldfであるかを確認
    closes_pldf = factory_closes_n4_1000()
    assert isinstance(closes_pldf, Pldf)
    ### get_labeled_datasetの検証 ###
    labeled_ds = closes_pldf.get_labeled_dataset()
    assert isinstance(labeled_ds, LabeledDataset)
    assert isinstance(labeled_ds.X, ndarray)
    assert isinstance(labeled_ds.y, ndarray)
    # Xの内容が正常に計算されてるか？
    assert (
        labeled_ds.X == closes_pldf.df[["lag_1", "lag_2", "lag_3", "lag_4"]].to_numpy()
    ).all()


def test_pldf_get_labeled_dataset_split():
    closes_pldf = factory_closes_n4_cycle()
    assert isinstance(closes_pldf, Pldf)
    lds_splt = closes_pldf.get_labeled_dataset_split()
    assert isinstance(lds_splt, LabeledDatasetSplit)
    # trainとtestが8:2に分割されてるか
    # WARN: factoryの内容に依存し過ぎたテスト
    assert lds_splt.train.y.shape[0] == 76
    assert lds_splt.test.y.shape[0] == 19


### LabeledDataset ###
def test_labeled_dataset():
    closes_pldf = factory_closes_n4_cycle()
    ld = closes_pldf.get_labeled_dataset()
    dataset = ld.to_lgb()
    assert isinstance(dataset, Dataset)


### LabeledDatasetSpli ###
def test_labeled_dataset_split():
    closes_pldf = factory_closes_n4_cycle()
    lds = closes_pldf.get_labeled_dataset_split()
    # to_lgb_train_test
    train, test = lds.get_lgb_train_test()
    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)
