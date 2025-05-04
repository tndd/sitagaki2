from fixture.infra.model.dataset import (
    DatasetImpl,
    DatasetImplV2,
    LabeledDatasetImpl,
    factory_dataset_impl,
    factory_dataset_impl_v2,
    factory_labeled_dataset_impl,
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


def test_factory_labeled_dataset_impl():
    ldsi: LabeledDatasetImpl = factory_labeled_dataset_impl()
    assert isinstance(ldsi, LabeledDatasetImpl)
    ldsi.pandera_schema.validate(ldsi.df)
    assert ldsi.columns == ["F0", "F1", "F2", "F3", "L"]
    assert ldsi.index == "Date"
    assert ldsi.label == "L"
