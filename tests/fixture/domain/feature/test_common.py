from fixture.domain.dataset.ohlcv import factory_ohlcv
from fixture.domain.feature.common import OhlcvFeatureImpl


def test_factory_ohlcv_feature_impl():
    ohlcv = factory_ohlcv()
    ohlcv_feature = OhlcvFeatureImpl(ohlcv)
    assert isinstance(ohlcv_feature, OhlcvFeatureImpl)
    assert ohlcv_feature.df.index.name == "Date"
    assert ohlcv_feature.df.index.dtype == "datetime64[ns]"
    assert ohlcv_feature.field.names == [
        "OF0",
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
    assert ohlcv_feature.field.label == ["OF0"]
    assert ohlcv_feature.field.exclude == ["OF5", "OF6"]
    assert ohlcv_feature.field.feature_names == ["OF1", "OF2", "OF3", "OF4"]
