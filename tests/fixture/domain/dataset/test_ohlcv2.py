from domain.dataset.ohlcv2 import Ohlcv2
from fixture.domain.dataset.ohlcv2 import factory_ohlcv2, factory_ohlcv2_random_walk


def test_factory_ohlcv2():
    ohlcv2 = factory_ohlcv2()
    assert isinstance(ohlcv2, Ohlcv2)


def test_factory_ohlcv2_random_walk():
    ohlcv2 = factory_ohlcv2_random_walk(n=1000)
    assert isinstance(ohlcv2, Ohlcv2)
    assert len(ohlcv2.df) == 1000
