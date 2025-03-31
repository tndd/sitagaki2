from domain.feature.lag import LagCloses10
from fixture.domain.dataset.ohlcv2 import factory_ohlcv2_random_walk


def factory_lag_closes10() -> LagCloses10:
    ohlcv = factory_ohlcv2_random_walk(n=1000)
    return LagCloses10(ohlcv)
