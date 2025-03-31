from lightgbm import Booster

from domain.model.lgbm import train_model_lgbm_lag_closes10
from fixture.domain.feature.lag import factory_lag_closes10


def test_train_model_lgbm_lag_closes10():
    lag_closes10 = factory_lag_closes10()
    model = train_model_lgbm_lag_closes10(lag_closes10)
    assert isinstance(model, Booster)
