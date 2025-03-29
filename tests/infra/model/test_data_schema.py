from fixture.infra.model import factory_data_schema_impl
from infra.model.data_schema import DataSchema
from infra.model.labeled_dataset import LabeledDataset, LabeledDatasetSplit


def test_data_schema():
    dsi = factory_data_schema_impl(
        label="Close",
        exclude="Volume",
    )
    # インスタンスが作成されてるか
    assert isinstance(dsi, DataSchema)
    # インデックスが設定されてるか
    assert dsi.df.index.name == "Date"
    assert dsi.df.index.dtype == "datetime64[ns]"
    # スキーマ検証の実行
    assert not dsi.SCHEMA.validate(dsi.df).empty
    # labelとexcludeがlistとして変換され設定されてるか
    assert dsi.label == ["Close"]
    assert dsi.exclude == ["Volume"]
    # test => get_col_names()
    assert dsi.get_col_names() == ["Open", "High", "Low", "Close", "Volume"]
    # test => get_labeled_dataset()
    assert isinstance(dsi.get_labeled_dataset(), LabeledDataset)
    # test => get_labeled_dataset_split()
    assert isinstance(dsi.get_labeled_dataset_split(), LabeledDatasetSplit)
