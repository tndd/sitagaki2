from domain.feature.closes.derive import derive_closes_n4
from domain.feature.closes.schema import ClosesN4
from fixture.factory.dataset.ohlcv import (
    factory_ohlcv,
    factory_ohlcv_1000,
    factory_ohlcv_cycle,
)


def factory_closes_n4() -> ClosesN4:
    ohlcv = factory_ohlcv()
    return derive_closes_n4(ohlcv)


def factory_closes_n4_1000() -> ClosesN4:
    ohlcv = factory_ohlcv_1000()
    return derive_closes_n4(ohlcv)


def factory_closes_n4_cycle() -> ClosesN4:
    ohlcv_cycle = factory_ohlcv_cycle()
    return derive_closes_n4(ohlcv_cycle)
