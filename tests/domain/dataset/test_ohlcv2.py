from domain.dataset.ohlcv2 import Ohlcv2
from fixture.domain.dataset.ohlcv2 import factory_ohlcv2


def test_factory_ohlcv2():
    ohlcv2 = factory_ohlcv2()
    assert isinstance(ohlcv2, Ohlcv2)
    assert ohlcv2.df.index.name == "Date"
    assert ohlcv2.df.index.dtype == "datetime64[ns]"
    assert ohlcv2.field.col_names == ["Open", "High", "Low", "Close", "Volume"]
