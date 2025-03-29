from domain.feature.closes import ClosesN4
from fixture.domain.feature.closes import (
    factory_closes_n4,
    factory_closes_n4_1000,
    factory_closes_n4_cycle,
)


def test_factory_closes_n4():
    closes_n4 = factory_closes_n4()
    assert isinstance(closes_n4, ClosesN4)
    assert closes_n4.df.schema == ClosesN4.SCHEMA


def test_factory_closes_n4_1000():
    closes_n4 = factory_closes_n4_1000()
    assert isinstance(closes_n4, ClosesN4)
    assert closes_n4.df.schema == ClosesN4.SCHEMA
    # データ数が1000から5件引かれた数
    assert closes_n4.df.height == (1000 - 5)


def test_factory_closes_n4_cycle():
    closes_n4 = factory_closes_n4_cycle()
    assert isinstance(closes_n4, ClosesN4)
    assert closes_n4.df.schema == ClosesN4.SCHEMA
