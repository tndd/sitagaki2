from fixture.infra.model.dataset import factory_labeled_dataset_impl
from infra.model.tensor import (
    LabeledTensor,
    SplitLabeledTensor,
    convert_labeled_dataset_to_split_tensor,
    convert_labeled_dataset_to_tensor,
)


def test_convert_labeled_dataset_to_tensor():
    ldsi = factory_labeled_dataset_impl()
    labeled_tensor = convert_labeled_dataset_to_tensor(ldsi)
    assert isinstance(labeled_tensor, LabeledTensor)
    assert labeled_tensor.X.shape == (10, 4)  # Lが除外されF0~F3の4カラムが10行
    assert labeled_tensor.y.shape == (10,)  # Lが目的変数に設定(10,)


def test_convert_labeled_dataset_to_split_tensor():
    ldsi = factory_labeled_dataset_impl()
    split_labeled_tensor = convert_labeled_dataset_to_split_tensor(ldsi)
    assert isinstance(split_labeled_tensor, SplitLabeledTensor)
