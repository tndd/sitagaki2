from feature.closes.schema import CLOSES_N4, CLOSES_N4_PLDF
from fixture.factory.feature.closes import factory_closes_n4


def test_factory_closes_n4():
    assert isinstance(factory_closes_n4(), CLOSES_N4)
    assert factory_closes_n4().schema == CLOSES_N4_PLDF.schema
