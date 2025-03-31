from domain.dataset.ohlcv import Ohlcv
from fixture.domain.dataset.ohlcv import factory_ohlcv


def test_factory_ohlcv():
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, Ohlcv)
    assert ohlcv.df.index.name == "Date"
    assert ohlcv.df.index.dtype == "datetime64[ns]"
    assert ohlcv.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
