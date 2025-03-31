from domain.feature.lag import LagCloses10
from fixture.domain.feature.lag import factory_lag_closes10


def test_factory_lag_closes10():
    lag_closes10 = factory_lag_closes10()
    assert isinstance(lag_closes10, LagCloses10)
    assert len(lag_closes10.df) == 990
    assert lag_closes10.field.index == "Date"
    assert lag_closes10.field.label == []
    assert lag_closes10.field.exclude == []
    assert lag_closes10.field.col_names == [
        "l0",
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
        "l6",
        "l7",
        "l8",
        "l9",
        "l10",
    ]
