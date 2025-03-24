from math import log

import pytest

from domain.feature.closes.derive import derive_closes_n4
from domain.feature.closes.schema import ClosesN4
from domain.feature.closes.service import calc_feature_closes
from domain.feature.common.const import SCALE_BP
from fixture.factory.dataset.ohlcv import factory_ohlcv_cycle


@pytest.mark.parametrize(
    "n_rows, n_lags, exp_height",
    [
        (10, 3, 6),  #      # 通常
        (6, 4, 1),  #       # ちょうど一行
        (4, 4, 0),  #       # 数が足りなければ0
        (3, 10, 0),  #       # 過剰な場合も0
    ],
)
def test_service_calc_feature_closes(
    n_rows,
    n_lags,
    exp_height,
):
    ohlcv = factory_ohlcv_cycle()
    df = ohlcv.df.head(n_rows)
    cf_closes = calc_feature_closes(df=df, n=n_lags)
    # 行数のチェック
    assert cf_closes.height == exp_height
    # n_lags + Date,now二件の合計
    assert cf_closes.width == n_lags + 2


def test_derive_closes_n4():
    # テストデータの準備
    ohlcv = factory_ohlcv_cycle()
    # テスト対象の関数を実行
    closes_n4 = derive_closes_n4(ohlcv)
    # スキーマの検証
    assert closes_n4.df.schema == ClosesN4.SCHEMA
    # 行数の検証 (100行入力 → 100 - 5 = 95行)
    assert closes_n4.df.shape == (95, 6)
    # excludeが設定されてるか？
    assert closes_n4.exclude == ["Date"]

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
