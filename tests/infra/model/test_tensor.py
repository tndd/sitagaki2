from fixture.infra.model.dataset import (
    DatasetImpl,
    factory_labeled_dataset_impl,
)
from infra.model.tensor import (
    LabeledTensor,
    SplitLabeledTensor,
    convert_labeled_dataset_to_split_tensor,
    convert_labeled_dataset_to_tensor,
)


def test_convert_labeled_dataset_to_tensor():
    ldsi: DatasetImpl = factory_labeled_dataset_impl()
    labeled_tensor = convert_labeled_dataset_to_tensor(ldsi)
    assert isinstance(labeled_tensor, LabeledTensor)


def test_convert_labeled_dataset_to_split_tensor():
    ldsi: DatasetImpl = factory_labeled_dataset_impl()
    split_labeled_tensor = convert_labeled_dataset_to_split_tensor(ldsi)
    assert isinstance(split_labeled_tensor, SplitLabeledTensor)
