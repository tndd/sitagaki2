from fixture.infra.model.dataset import (
    DatasetImpl,
    DatasetImplV2,
    factory_dataset_impl,
    factory_dataset_impl_v2,
)


def test_factory_dataset_impl():
    dsi: DatasetImpl = factory_dataset_impl()
    assert isinstance(dsi, DatasetImpl)
    dsi.field.schema.validate(dsi.df)
    assert dsi.df.index.name == "Date"
    assert dsi.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
    assert dsi.field.label == []
    assert dsi.field.exclude == []


def test_factory_dataset_impl_v2():
    dsi: DatasetImplV2 = factory_dataset_impl_v2()
    assert isinstance(dsi, DatasetImplV2)
    dsi.field.schema.validate(dsi.df)
    assert dsi.df.index.name == "Date"
    assert dsi.field.col_names == ["D0", "D1", "D2", "D3", "D4"]
    assert dsi.field.label == []
    assert dsi.field.exclude == []
