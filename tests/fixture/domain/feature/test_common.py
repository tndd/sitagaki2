from fixture.domain.feature.common import (
    OhlcvFeatureImpl,
    factory_ohlcv_feature_impl,
)


def test_factory_ohlcv_feature_impl():
    ohlcv_feature = factory_ohlcv_feature_impl()
    assert isinstance(ohlcv_feature, OhlcvFeatureImpl)
