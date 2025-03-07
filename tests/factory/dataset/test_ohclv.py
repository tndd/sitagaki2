from datetime import date

from dataset.schema import OHLCV, OHLCV_PLDF
from fixture.factory.dataset.ohclv import factory_ohclv


def test_factory_ohlcv():
    ohclv = factory_ohclv()
    assert isinstance(ohclv, OHLCV)
    # データの検証
    assert len(ohclv) == 4  # データ件数が4件であることを確認
    # スキーマの完全一致チェック
    assert ohclv.schema == OHLCV_PLDF.schema
    # Dateが日付順にソートされていること"
    assert ohclv["Date"].is_sorted()
    # 日付の内容確認
    assert ohclv["Date"].to_list() == [
        date(2000, 1, 1),
        date(2000, 1, 2),
        date(2000, 1, 3),
        date(2000, 1, 4),
    ]
    # Openの内容確認
    assert ohclv["Open"].to_list() == [100.0, 200.0, 300.0, 400.0]
    # Volumeの内容確認
    assert ohclv["Volume"].to_list() == [1000, 1100, 1200, 1300]
