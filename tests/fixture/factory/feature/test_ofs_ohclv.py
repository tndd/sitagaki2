from feature.schema import OFS_OHCLV
from fixture.factory.feature.ofs_ohclv import factory_ofs_ohclv


def test_factory_ofs_ohclv():
    ofs_ohlcv = factory_ofs_ohclv()
    assert isinstance(ofs_ohlcv, OFS_OHCLV)
