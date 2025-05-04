from pandera import DataFrameSchema

from fixture.infra.model.dataset import (
    factory_dataset_impl,
    factory_labeled_dataset_impl,
)
from infra.model.draft import Dataset, LabeledDataset


def test_dataset():
    dsi = factory_dataset_impl()
    assert isinstance(dsi, Dataset)
    # pandera
    assert isinstance(dsi.pandera_schema, DataFrameSchema)
    # columns
    assert isinstance(dsi.columns, list)
    assert all(isinstance(col, str) for col in dsi.columns)


def test_labeled_dataset():
    ldsi = factory_labeled_dataset_impl()
    assert isinstance(ldsi, LabeledDataset)
    # non_label_columns
    assert isinstance(ldsi.non_label_columns, list)
    assert all(isinstance(col, str) for col in ldsi.non_label_columns)
