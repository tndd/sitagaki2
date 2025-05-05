from pandera import DataFrameSchema

from fixture.infra.model.dataset import (
    factory_dataset_impl,
    factory_labeled_dataset_impl,
)
from infra.model.dataset import Dataset, LabeledDataset


def test_dataset():
    dataset = factory_dataset_impl()
    assert isinstance(dataset, Dataset)
    # pandera
    assert isinstance(dataset.pandera_schema, DataFrameSchema)
    # columns
    assert isinstance(dataset.columns, list)
    assert all(isinstance(col, str) for col in dataset.columns)


def test_dataset_undefined_column():
    """
    未定義のカラムを持つdfを渡した場合の挙動のテスト
    """
    # TODO: 実装
    pass


def test_labeled_dataset():
    labeled_dataset = factory_labeled_dataset_impl()
    assert isinstance(labeled_dataset, LabeledDataset)
    # non_label_columns
    assert isinstance(labeled_dataset.non_label_columns, list)
    assert all(isinstance(col, str) for col in labeled_dataset.non_label_columns)
