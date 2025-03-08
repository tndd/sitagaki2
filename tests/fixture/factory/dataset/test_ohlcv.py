from datetime import datetime

from dataset.ohlcv.schema import OHLCV, OHLCV_PLDF
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
    # スキーマの完全一致チェック
    assert ohlcv.schema == OHLCV_PLDF.schema
    # Dateが日付順にソートされていること"
    assert ohlcv["Date"].is_sorted()
    # 値の順序関係は妥当か？
    ## Highは必ず最も高い値
    assert (ohlcv["High"] >= ohlcv["Open"]).all()
    assert (ohlcv["High"] >= ohlcv["Close"]).all()
    ## かつLowは最も低い値
    assert (ohlcv["High"] >= ohlcv["Low"]).all()
    assert (ohlcv["Open"] >= ohlcv["Low"]).all()
    assert (ohlcv["Close"] >= ohlcv["Low"]).all()

    ### 具体的な値のチェック ###
    # データ件数が4件であることを確認
    assert len(ohlcv) == 4
    # 日付の内容確認
    assert ohlcv["Date"].to_list() == [
        datetime(2000, 1, 1),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
    ]
    # Openの内容確認 (100ずつ増える。ただし最後は101.0でCloseと同じ値)
    assert ohlcv["Open"].to_list() == [100.0, 200.0, 300.0, 101.0]
    # Highの内容確認を追加 (Openを基準に5%ずつ日毎に上昇幅が上昇)
    assert ohlcv["High"].to_list() == [105.0, 210.0, 345.0, 120.0]
    # Lowの内容確認を追加 (Openを基準に5%ずつ日毎に下落幅が下落)
    assert ohlcv["Low"].to_list() == [95.0, 190.0, 255.0, 80.0]
    # Closeの内容確認を追加 (Openを基準に1%ずつ上昇幅が上昇。ただし最後はOpenと同じ価格となる)
    assert ohlcv["Close"].to_list() == [101.0, 204.0, 309.0, 101.0]
    # Volumeの内容確認 (1000から日毎に100ずつ上昇。ただし最終日は初めと同じ値)
    assert ohlcv["Volume"].to_list() == [1000, 1100, 1200, 1000]
