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
    # デフォルト指定での分割では、テストサイズは20%。データ数は10行。
    split_labeled_tensor = convert_labeled_dataset_to_split_tensor(ldsi)
    assert isinstance(split_labeled_tensor, SplitLabeledTensor)
    # trainは8行/testは２行。カラム数はLを除く4カラム
    assert split_labeled_tensor.train.X.shape == (8, 4)
    assert split_labeled_tensor.test.X.shape == (2, 4)
    # trainは8行/testは2行。ラベルはLのみなので1カラム。
    assert split_labeled_tensor.train.y.shape == (8,)
    assert split_labeled_tensor.test.y.shape == (2,)
