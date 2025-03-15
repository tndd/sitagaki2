from domain.feature.closes.schema import ClosesN4
from fixture.factory.feature.closes import factory_closes_n4


def test_factory_closes_n4():
    closes_n4 = factory_closes_n4()
    assert isinstance(closes_n4, ClosesN4)
    assert closes_n4.df.schema == ClosesN4.SCHEMA
