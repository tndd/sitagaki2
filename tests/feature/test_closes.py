from math import log

import pytest

from common.const import SCALE_BP
from feature.closes.derive import derive_closes_n4
from feature.closes.schema import CLOSES_N4_PLDF
from fixture.factory.dataset.ohlcv import factory_ohlcv_cycle


def test_derive_closes_n4():
    # テストデータの準備
    df = factory_ohlcv_cycle()
    # テスト対象の関数を実行
    result = derive_closes_n4(df)
    # スキーマの検証
    assert result.schema == CLOSES_N4_PLDF.schema
    # 行数の検証 (100行入力 → 100 - 5 = 95行)
    assert result.shape == (95, 6)

    ### 始行の内容チェック　###
    first_row = result.row(0, named=True)
    # 入力データから期待される値を計算
    close_values = df.get_column("Close").to_list()
    # 各カラムの期待値計算
    expected_now = log(close_values[5] / close_values[4]) * SCALE_BP
    expected_lag1 = log(close_values[4] / close_values[3]) * SCALE_BP
    expected_lag2 = log(close_values[3] / close_values[2]) * SCALE_BP
    expected_lag3 = log(close_values[2] / close_values[1]) * SCALE_BP
    expected_lag4 = log(close_values[1] / close_values[0]) * SCALE_BP
    # assert
    assert first_row["now"] == pytest.approx(expected_now, abs=1e-9)
    assert first_row["lag_1"] == pytest.approx(expected_lag1, abs=1e-9)
    assert first_row["lag_2"] == pytest.approx(expected_lag2, abs=1e-9)
    assert first_row["lag_3"] == pytest.approx(expected_lag3, abs=1e-9)
    assert first_row["lag_4"] == pytest.approx(expected_lag4, abs=1e-9)
    assert first_row["lag_4"] == pytest.approx(expected_lag4, abs=1e-9)

    ### 差分の伝播チェック ###
    
