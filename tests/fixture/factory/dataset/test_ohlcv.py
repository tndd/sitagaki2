from datetime import date

from dataset.schema import OHLCV, OHLCV_PLDF
from fixture.factory.dataset.ohlcv import factory_ohlcv


def test_factory_ohlcv():
    """
    MEMO: fixtureにも関わらず詳細なテストを行なってる理由
        ここは本体部分の処理というよりもfactoryが、
        要求される値でオブジェクトを作ってるか？という観点でテストされてるから。
        そうでもなければfixtureでこれほど長いテストを行うのは規約違反だ。
    """
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, OHLCV)
    # データの検証
    assert len(ohlcv) == 4  # データ件数が4件であることを確認
    # スキーマの完全一致チェック
    assert ohlcv.schema == OHLCV_PLDF.schema
    # Dateが日付順にソートされていること"
    assert ohlcv["Date"].is_sorted()
    # 日付の内容確認
    assert ohlcv["Date"].to_list() == [
        date(2000, 1, 1),
        date(2000, 1, 2),
        date(2000, 1, 3),
        date(2000, 1, 4),
    ]
    # Openの内容確認
    assert ohlcv["Open"].to_list() == [100.0, 200.0, 300.0, 100.0]
    # Highの内容確認を追加
    assert ohlcv["High"].to_list() == [105.0, 210.0, 345.0, 120.0]
    # Lowの内容確認を追加
    assert ohlcv["Low"].to_list() == [95.0, 190.0, 255.0, 80.0]
    # Closeの内容確認を追加
    assert ohlcv["Close"].to_list() == [101.0, 204.0, 309.0, 100.0]
    # Volumeの内容確認
    assert ohlcv["Volume"].to_list() == [1000, 1100, 1200, 1000]
