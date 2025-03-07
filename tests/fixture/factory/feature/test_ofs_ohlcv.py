from feature.schema import OFS_OHLCV
from fixtures.factory.feature.ofs_ohlcv import factory_ofs_ohlcv


def test_factory_ofs_ohlcv():
    ofs_ohlcv = factory_ofs_ohlcv()
    assert isinstance(ofs_ohlcv, OFS_OHLCV)
    # TODO: ofs_ohlcvの内容確認処理
