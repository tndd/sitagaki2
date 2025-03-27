from datetime import datetime

from domain.dataset.ohlcv import Ohlcv
from fixture.factory.dataset.ohlcv import (
    factory_ohlcv,
    factory_ohlcv_1000,
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
    assert ohlcv.df.schema == Ohlcv.SCHEMA
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


def test_factory_ohlcv_1000():
    ohlcv = factory_ohlcv_1000()

    assert isinstance(ohlcv, Ohlcv)
    assert ohlcv.df.schema == Ohlcv.SCHEMA
    assert ohlcv.df.height == 1000

    # head5の期待される値
    expected_data = [
        {
            "Date": datetime(2023, 1, 1),
            "Open": 137.454,
            "High": 139.305,
            "Low": 134.837,
            "Close": 139.181,
            "Volume": 5713,
        },
        {
            "Date": datetime(2023, 1, 2),
            "Open": 195.071,
            "High": 200.49,
            "Low": 192.601,
            "Close": 198.038,
            "Volume": 4092,
        },
        {
            "Date": datetime(2023, 1, 3),
            "Open": 173.199,
            "High": 181.928,
            "Low": 164.136,
            "Close": 170.704,
            "Volume": 5159,
        },
        {
            "Date": datetime(2023, 1, 4),
            "Open": 159.866,
            "High": 167.188,
            "Low": 157.371,
            "Close": 161.115,
            "Volume": 7308,
        },
        {
            "Date": datetime(2023, 1, 5),
            "Open": 115.602,
            "High": 123.668,
            "Low": 112.883,
            "Close": 116.319,
            "Volume": 8478,
        },
    ]
    # 検証元のデータからheadの５行取得して検証
    actual_data = ohlcv.df.head(5).rows(named=True)
    for expected, actual in zip(expected_data, actual_data, strict=True):
        assert expected == actual


def test_factory_ohlcv_cycle():
    ohlvc_cycle = factory_ohlcv_cycle()
    assert isinstance(ohlvc_cycle, Ohlcv)


def test_factory_ohlcv_brown():
    ohlcv = factory_ohlcv_brown()
    assert isinstance(ohlcv, Ohlcv)
