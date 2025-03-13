from datetime import datetime

from dataset.ohlcv.schema import Ohlcv
from fixture.factory.dataset.ohlcv import (
    factory_ohlcv,
    factory_ohlcv_brown,
    factory_ohlcv_cycle,
)


def test_factory_ohlcv():
    """
    MEMO: fixtureにも関わらず詳細なテストを行なってる理由
        ここは本体部分の処理というよりもfactoryが、
        要求される値でオブジェクトを作ってるか？という観点でテストされてるから。
        そうでもなければfixtureでこれほど長いテストを行うのは規約違反だ。
    """
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, Ohlcv)
    # スキーマの完全一致チェック
    assert ohlcv.df.schema == Ohlcv.schema
    # Dateが日付順にソートされていること"
    assert ohlcv.df["Date"].is_sorted()
    # 値の順序関係は妥当か？
    ## Highは必ず最も高い値
    assert (ohlcv.df["High"] >= ohlcv.df["Open"]).all()
    assert (ohlcv.df["High"] >= ohlcv.df["Close"]).all()
    ## かつLowは最も低い値
    assert (ohlcv.df["High"] >= ohlcv.df["Low"]).all()
    assert (ohlcv.df["Open"] >= ohlcv.df["Low"]).all()
    assert (ohlcv.df["Close"] >= ohlcv.df["Low"]).all()

    ### 具体的な値のチェック ###
    # データ件数が4件であることを確認
    assert len(ohlcv.df) == 4
    # 日付の内容確認
    assert ohlcv.df["Date"].to_list() == [
        datetime(2000, 1, 1),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
    ]
    # Openの内容確認 (100ずつ増える。ただし最後は101.0でCloseと同じ値)
    assert ohlcv.df["Open"].to_list() == [100.0, 200.0, 300.0, 101.0]
    # Highの内容確認を追加 (Openを基準に5%ずつ日毎に上昇幅が上昇)
    assert ohlcv.df["High"].to_list() == [105.0, 210.0, 345.0, 120.0]
    # Lowの内容確認を追加 (Openを基準に5%ずつ日毎に下落幅が下落)
    assert ohlcv.df["Low"].to_list() == [95.0, 190.0, 255.0, 80.0]
    # Closeの内容確認を追加 (Openを基準に1%ずつ上昇幅が上昇。ただし最後はOpenと同じ価格となる)
    assert ohlcv.df["Close"].to_list() == [101.0, 204.0, 309.0, 101.0]
    # Volumeの内容確認 (1000から日毎に100ずつ上昇。ただし最終日は初めと同じ値)
    assert ohlcv.df["Volume"].to_list() == [1000, 1100, 1200, 1000]


def test_factory_ohlcv_cycle():
    ohlvc_cycle = factory_ohlcv_cycle()
    assert isinstance(ohlvc_cycle, Ohlcv)


def test_factory_ohlcv_brown():
    ohlcv = factory_ohlcv_brown()
    assert isinstance(ohlcv, Ohlcv)
