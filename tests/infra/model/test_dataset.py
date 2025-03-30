from fixture.infra.model.dataset import factory_dataset_impl
from infra.model.dataset import Dataset
from infra.model.tensor import LabeledTensor, SplitLabeledTensor


def test_dataset():
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
    assert dsi.field.label == ["Close"]
    assert dsi.field.exclude == ["Volume"]
    # カラム名の確認
    assert dsi.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
    # test => get_labeled_tensor()
    labeled_tensor = dsi.get_labeled_tensor()
    assert isinstance(labeled_tensor, LabeledTensor)
    assert labeled_tensor.X.shape == (9, 3)  # CloseとVolumeが除外(9,3)
    assert labeled_tensor.y.shape == (9,)  # Closeが目的変数に設定(9,)
    # Open, High, Lowの値が設定されているか
    assert labeled_tensor.X.tolist() == [
        [100.0, 110.0, 90.0],
        [200.0, 210.0, 190.0],
        [300.0, 310.0, 290.0],
        [400.0, 410.0, 390.0],
        [500.0, 510.0, 490.0],
        [600.0, 610.0, 590.0],
        [700.0, 710.0, 690.0],
        [800.0, 810.0, 790.0],
        [900.0, 910.0, 890.0],
    ]
    # Closeが目的変数に設定されているか
    assert labeled_tensor.y.tolist() == [
        105.0,
        205.0,
        305.0,
        405.0,
        505.0,
        605.0,
        705.0,
        805.0,
        905.0,
    ]
    # test => get_split_labeled_tensor()
    sl_tensor = dsi.get_split_labeled_tensor()
    assert isinstance(sl_tensor, SplitLabeledTensor)
    assert sl_tensor.train.X.shape == (7, 3)  # 7つのデータが訓練用に分割される
    assert sl_tensor.test.X.shape == (2, 3)  # 2つのデータがテスト用に分割される
    assert sl_tensor.train.y.shape == (7,)  # 7つのデータが訓練用に分割される
    assert sl_tensor.test.y.shape == (2,)  # 2つのデータがテスト用に分割される
