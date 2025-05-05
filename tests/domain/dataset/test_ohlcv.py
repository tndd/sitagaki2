from domain.dataset.ohlcv import Ohlcv
from fixture.domain.dataset.ohlcv import factory_ohlcv


def test_factory_ohlcv():
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, Ohlcv)
    assert ohlcv.index == "Date"
    assert ohlcv.df.index.dtype == "datetime64[ns]"
    assert ohlcv.columns == ["Open", "High", "Low", "Close", "Volume"]
