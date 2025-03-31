from domain.feature.lag import LagCloses10
from fixture.domain.feature.lag import factory_lag_closes10


def test_factory_lag_closes10():
    lag_closes10 = factory_lag_closes10()
    assert isinstance(lag_closes10, LagCloses10)
    assert len(lag_closes10.df) == 989  # 0 ~ 10日分のデータが欠損するため
