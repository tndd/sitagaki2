from fixture.infra.model.dataset import (
    DatasetImpl,
    DatasetImplV2,
    factory_dataset_impl,
    factory_dataset_impl_v2,
)


def test_factory_dataset_impl():
    dsi: DatasetImpl = factory_dataset_impl()
    assert isinstance(dsi, DatasetImpl)
    dsi.pandera_schema.validate(dsi.df)
    assert dsi.columns == ["Open", "High", "Low", "Close", "Volume"]
    assert dsi.index == "Date"


def test_factory_dataset_impl_v2():
    dsi: DatasetImplV2 = factory_dataset_impl_v2()
    assert isinstance(dsi, DatasetImplV2)
    dsi.pandera_schema.validate(dsi.df)
    assert dsi.columns == ["D0", "D1", "D2", "D3", "D4"]
    assert dsi.index == "Date"
