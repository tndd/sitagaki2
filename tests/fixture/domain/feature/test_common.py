from fixture.domain.dataset.ohlcv import factory_ohlcv
from fixture.domain.feature.common import OhlcvFeatureImpl


def test_factory_ohlcv_feature_impl():
    """
    OhlcvFeatureImplのファクトリのテスト
    テスト内容はインスタンスが作れているか最低限のみ
    """
    ohlcv = factory_ohlcv()
    ohlcv_feature = OhlcvFeatureImpl(ohlcv)
    assert isinstance(ohlcv_feature, OhlcvFeatureImpl)
