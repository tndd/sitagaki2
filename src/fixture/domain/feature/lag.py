from domain.feature.lag import LagCloses10
from fixture.domain.dataset.ohlcv import factory_ohlcv_random_walk


def factory_lag_closes10(n: int = 1000) -> LagCloses10:
    ohlcv = factory_ohlcv_random_walk(n)
    return LagCloses10(ohlcv)
