from fixture.infra.model import factory_dataset_impl
from infra.model.dataset import Dataset
from infra.model.tensor import LabeledTensor, SplitLabeledTensor


def test_data_schema():
    dsi = factory_dataset_impl(
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
    # test => get_labeled_tensor()
    labeled_tensor = dsi.get_labeled_tensor()
    assert isinstance(labeled_tensor, LabeledTensor)
    assert labeled_tensor.X.shape == (3, 3)  # CloseとVolumeが除外(3,3)
    assert labeled_tensor.y.shape == (3,)  # Closeが目的変数に設定(3,)
    assert labeled_tensor.X.tolist() == [
        [101.0, 102.0, 103.0],  # Open
        [201.0, 202.0, 203.0],  # High
        [301.0, 302.0, 303.0],  # Low
    ]
    assert labeled_tensor.y.tolist() == [104.0, 204.0, 304.0]  # Closeが目的変数に設定
    # test => get_split_labeled_tensor()
    sl_tensor = dsi.get_split_labeled_tensor()
    assert isinstance(sl_tensor, SplitLabeledTensor)
    assert sl_tensor.train.X.shape == (2, 3)  # 2つのデータが訓練用に分割される
    assert sl_tensor.test.X.shape == (1, 3)  # 1つのデータがテスト用に分割される
    assert sl_tensor.train.y.shape == (2,)  # 2つのデータが訓練用に分割される
    assert sl_tensor.test.y.shape == (1,)  # 1つのデータがテスト用に分割される
