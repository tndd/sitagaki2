from lightgbm import Booster

from domain.model.lgbm import train_model_lgbm_ohlcv_feature
from fixture.domain.feature.lag import factory_lag_closes10


def test_train_model_lgbm_ohlcv_feature():
    # lag_closes10
    lag_closes = factory_lag_closes10()
    model = train_model_lgbm_ohlcv_feature(lag_closes)
    assert isinstance(model, Booster)
