from math import log

from domain.feature.closes.derive import derive_closes_n4
from domain.feature.closes.schema import ClosesN4
from domain.feature.common.const import SCALE_BP
from fixture.factory.dataset.ohlcv import factory_ohlcv_cycle


def test_derive_closes_n4():
    # テストデータの準備
    ohlcv = factory_ohlcv_cycle()
    # テスト対象の関数を実行
    closes_n4 = derive_closes_n4(ohlcv)
    # スキーマの検証
    assert closes_n4.df.schema == ClosesN4.SCHEMA
    # 行数の検証 (100行入力 → 100 - 5 = 95行)
    assert closes_n4.df.shape == (95, 6)

    ### 始行の内容チェック　###
    first_row = closes_n4.df.row(0, named=True)
    # 入力データから期待される値を計算
    close_values = ohlcv.df.get_column("Close").to_list()
    assert first_row["now"] == log(close_values[5] / close_values[4]) * SCALE_BP
    assert first_row["lag_1"] == log(close_values[4] / close_values[3]) * SCALE_BP
    assert first_row["lag_2"] == log(close_values[3] / close_values[2]) * SCALE_BP
    assert first_row["lag_3"] == log(close_values[2] / close_values[1]) * SCALE_BP
    assert first_row["lag_4"] == log(close_values[1] / close_values[0]) * SCALE_BP

    ### 差分の伝播(shift)チェック ###
    rows = closes_n4.df.rows(named=True)
    # 3ステップ分、値がshiftしてるか？
    for i in range(3):
        assert rows[i]["now"] == rows[i + 1]["lag_1"]
        assert rows[i]["lag_1"] == rows[i + 1]["lag_2"]
        assert rows[i]["lag_2"] == rows[i + 1]["lag_3"]
        assert rows[i]["lag_3"] == rows[i + 1]["lag_4"]
