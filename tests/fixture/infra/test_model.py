from fixture.infra.model.dataset import DatasetImpl, factory_dataset_impl
from fixture.infra.model.feature import FeatureImpl, factory_feature_impl


def test_factory_dataset_impl():
    dsi: DatasetImpl = factory_dataset_impl()
    assert isinstance(dsi, DatasetImpl)
    dsi.field.schema.validate(dsi.df)
    assert dsi.df.index.name == "Date"
    assert dsi.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
    assert dsi.field.label == []
    assert dsi.field.exclude == []


def test_factory_feature_impl():
    feature: FeatureImpl = factory_feature_impl()
    assert isinstance(feature, FeatureImpl)
    feature.field.schema.validate(feature.df)
    assert feature.df.index.name == "Date"
    assert feature.field.col_names == ["f0", "f1", "f2", "f3", "f4"]
    assert feature.field.label == ["f0"]
    assert feature.field.exclude == []
