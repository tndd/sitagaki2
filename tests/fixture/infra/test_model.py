from fixture.infra.model.dataset import (
    DatasetImpl,
    DatasetImplV2,
    LabeledDatasetImpl,
    factory_dataset_impl,
    factory_dataset_impl_v2,
    factory_labeled_dataset_impl,
)


def test_factory_dataset_impl():
    dataset: DatasetImpl = factory_dataset_impl()
    assert isinstance(dataset, DatasetImpl)
    dataset.pandera_schema.validate(dataset.df)
    assert dataset.columns == ["Open", "High", "Low", "Close", "Volume"]
    assert dataset.index == "Date"


def test_factory_dataset_impl_v2():
    dataset_v2: DatasetImplV2 = factory_dataset_impl_v2()
    assert isinstance(dataset_v2, DatasetImplV2)
    dataset_v2.pandera_schema.validate(dataset_v2.df)
    assert dataset_v2.columns == ["D0", "D1", "D2", "D3", "D4"]
    assert dataset_v2.index == "Date"


def test_factory_labeled_dataset_impl():
    labeled_dataset: LabeledDatasetImpl = factory_labeled_dataset_impl()
    assert isinstance(labeled_dataset, LabeledDatasetImpl)
    labeled_dataset.pandera_schema.validate(labeled_dataset.df)
    assert labeled_dataset.columns == ["F0", "F1", "F2", "F3", "L"]
    assert labeled_dataset.index == "Date"
    assert labeled_dataset.label == "L"
