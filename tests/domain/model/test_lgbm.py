from lightgbm import Booster

from domain.model.lgbm import train_model_lgbm_closes_n4
from fixture.factory.feature.closes import factory_closes_n4_cycle


def test_train_model_lgbm_closes_n4():
    closes = factory_closes_n4_cycle()
    model = train_model_lgbm_closes_n4(closes)
    assert isinstance(model, Booster)
