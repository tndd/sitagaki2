from feature.closes.derive import derive_closes_n4
from feature.closes.schema import CLOSES_N4
from fixture.factory.dataset.ohlcv import factory_ohlcv_cycle


def factory_closes_n4() -> CLOSES_N4:
    df = factory_ohlcv_cycle()
    return derive_closes_n4(df)
