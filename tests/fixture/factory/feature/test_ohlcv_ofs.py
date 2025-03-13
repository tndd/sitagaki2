from feature.ohlcv_ofs.schema import OhlcvOfs
from fixture.factory.feature.ohlcv_ofs import factory_ofs_ohlcv


def test_factory_ofs_ohlcv():
    """
    MEMO: factory ohlcvのテスト内容について
        ofs_ohlcvの内容については、derive_ofs_ohlcvの方で検証する。
        factoryの方は、インスタンスができてるかどうか？という最低限のテスト。

    NOTE: fixtureのテスト実装方針
        fixtureの基本方針として、fixtureについてのテストは最低限とする。
        機能の品質は、各関数ごとのテストで担保されるべきという思想。
        二重テスト状態はなるべく避けたい。
    """
    ofs_ohlcv = factory_ofs_ohlcv()
    assert isinstance(ofs_ohlcv, OhlcvOfs)
