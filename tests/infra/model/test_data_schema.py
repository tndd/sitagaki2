from fixture.infra.model import factory_data_schema_impl
from infra.model.dataset import Dataset
from infra.model.labeled_dataset import LabeledDataset, LabeledDatasetSplit


def test_data_schema():
    dsi = factory_data_schema_impl(
        label="Close",
        exclude="Volume",
    )
    # インスタンスが作成されてるか
    assert isinstance(dsi, Dataset)
    # インデックスが設定されてるか
    assert dsi.df.index.name == "Date"
    assert dsi.df.index.dtype == "datetime64[ns]"
    # スキーマ検証の実行 (DataSchema内部で検証はされてるが、念のため)
    assert not dsi.field.schema.validate(dsi.df).empty
    # labelとexcludeがlistとして変換され設定されてるか
    assert dsi.label == ["Close"]
    assert dsi.exclude == ["Volume"]
    # カラム名の確認
    assert dsi.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
    # test => get_labeled_dataset()
    lds = dsi.get_labeled_dataset()
    assert isinstance(lds, LabeledDataset)
    assert lds.X.shape == (3, 3)  # CloseとVolumeが除外(3,3)
    assert lds.y.shape == (3,)  # Closeが目的変数に設定(3,)
    assert lds.X.tolist() == [
        [101.0, 102.0, 103.0],  # Open
        [201.0, 202.0, 203.0],  # High
        [301.0, 302.0, 303.0],  # Low
    ]
    assert lds.y.tolist() == [104.0, 204.0, 304.0]  # Closeが目的変数に設定
    # test => get_labeled_dataset_split()
    ldsv = dsi.get_labeled_dataset_split()
    assert isinstance(ldsv, LabeledDatasetSplit)
    assert ldsv.train.X.shape == (2, 3)  # 2つのデータが訓練用に分割される
    assert ldsv.test.X.shape == (1, 3)  # 1つのデータがテスト用に分割される
    assert ldsv.train.y.shape == (2,)  # 2つのデータが訓練用に分割される
    assert ldsv.test.y.shape == (1,)  # 1つのデータがテスト用に分割される
