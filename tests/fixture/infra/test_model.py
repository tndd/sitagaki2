from fixture.infra.model.dataset import DatasetImpl, factory_dataset_impl
from fixture.infra.model.feature import FeatureImpl, factory_feature_impl


def test_factory_dataset_impl():
    dsi: DatasetImpl = factory_dataset_impl()
    assert isinstance(dsi, DatasetImpl)
    assert dsi.df.index.name == "Date"


def test_factory_feature_impl():
    feature: FeatureImpl = factory_feature_impl()
    assert isinstance(feature, FeatureImpl)
    assert feature.df.index.name == "Date"
