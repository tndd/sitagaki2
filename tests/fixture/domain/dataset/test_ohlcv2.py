from domain.dataset.ohlcv2 import Ohlcv2
from fixture.domain.dataset.ohlcv2 import factory_ohlcv2


def test_factory_ohlcv2():
    ohlcv2 = factory_ohlcv2()
    assert isinstance(ohlcv2, Ohlcv2)
