from domain.dataset.ohlcv import Ohlcv
from fixture.domain.dataset.ohlcv import factory_ohlcv, factory_ohlcv_random_walk


def test_factory_ohlcv():
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, Ohlcv)


def test_factory_ohlcv_random_walk():
    ohlcv = factory_ohlcv_random_walk(n=1000)
    assert isinstance(ohlcv, Ohlcv)
    assert len(ohlcv.df) == 1000
